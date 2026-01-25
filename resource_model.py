import os

import pandas as pd

from tqdm.contrib.concurrent import process_map
from pywr.model import Model
from pywr.nodes import Catchment, Link, Output, Storage, RiverGauge
from pywr.parameters import DataFrameParameter, MonthlyProfileParameter, ControlCurveIndexParameter, IndexedArrayParameter, AggregatedParameter
from pywr.recorders import NumpyArrayStorageRecorder, NumpyArrayNodeRecorder, NumpyArrayParameterRecorder

from pathlib import Path

def run_resource_model(flows_path: Path, controlled: bool = False) -> pd.DataFrame:
    flows = pd.read_csv(flows_path)

    dates = pd.date_range("2026-12-01", "2099-11-30")
    flows.index = dates

    flows = flows.loc["2029-12-01":]

    flows["discharge"] = flows["discharge at the outlet - regular timestep   24.00 hours"] * 3600 * 24 / 1000

    model = Model()

    model.timestepper.start = pd.Timestamp("2029-12-01")
    model.timestepper.end = pd.Timestamp("2099-11-30")
    model.timestepper.timestep = 1

    public_demand_profile = MonthlyProfileParameter(
        model,
        values=[40.0, 40.0, 50.0, 60.0, 60.0, 80.0, 80.0, 80.0, 60.0, 50.0, 40.0, 40.0],
        name="Public Demand Profile"
    )

    controlled_release_cost_profile = MonthlyProfileParameter(
        model,
        values=[0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 0.0, 0.0, 0.0],
        name="Controlled Release Cost Profile"
    )

    catchment_1_flow = DataFrameParameter(
        model,
        flows["discharge"],
        name="Catchment 1 Flows"
    )

    abstraction_junction = Link(
        model,
        "abstraction_junction",
        position=(42.1368, 49.6409)
    )

    catchment_1 = Catchment(
        model,
        "catchment_1",
        flow= catchment_1_flow,
        position=(41.987, 22.4417)
    )

    compensation = Link(
        model,
        "compensation",
        position=(42.0388, 40.6013),
        cost= -10,
        max_flow = 50.0,
        min_flow = 0.0
    )

    controlled_release = Link(
        model,
        "controlled_release",
        max_flow=120.0,
        cost=controlled_release_cost_profile,
        position=(41.2212, 40.5896)
    )

    nonpublic_demand = Output(
        model,
        "nonpublic_demand",
        cost= -5.0,
        max_flow= 10.0,
        position= (49.7254, 49.9042)
    )

    outflow = Output(
        model,
        "outflow",
        position= (42.1368, 70.1077)
    )

    storage_reservoir = Storage(
        model,
        "storage_reservoir",
        cost= 0,
        min_volume= 0.0,
        max_volume= 46345.0,
        initial_volume= 46345.0,
        position= (42.0903, 31.8102)
    )

    if controlled:
        control_curve_lower = MonthlyProfileParameter(
            model,
            values=[0.5] * 12,
            name="Control Curve Lower (50%)"
        )
        
        control_curve_middle = MonthlyProfileParameter(
            model,
            values=[0.8] * 12,
            name="Control Curve Middle (80%)"
        )

        control_curve_index = ControlCurveIndexParameter(
            model,
            storage_reservoir,
            [control_curve_middle, control_curve_lower],
            name="Control Curve Index"
        )

        demand_saving_factor = IndexedArrayParameter(
            model,
            control_curve_index,
            [1.0, 0.85, 0.6],
            name="demand_saving_factor"
        )

        public_demand_max_flow = AggregatedParameter(
            model,
            [public_demand_profile, demand_saving_factor],
            agg_func="product",
            name="Restricted Public Demand"
        )
    else:
        public_demand_max_flow = public_demand_profile

    public_demand = Output(
        model,
        "public_demand",
        max_flow= public_demand_max_flow,
        cost= -10.0,
        position= (32.8008, 49.8205)
    )

    rivergauge = RiverGauge(
        model= model,
        name= "rivergauge",
        cost= 0.0,
        mrf_cost = -10.0,
        mrf= 50.0,
        position= (42.2041, 59.5685)
    )

    spill = Link(
        model,
        "spill",
        cost= 5000,
        position= (42.9557, 40.5319)
    )

    catchment_1.connect(storage_reservoir)

    storage_reservoir.connect(compensation)
    storage_reservoir.connect(controlled_release)
    storage_reservoir.connect(spill)

    compensation.connect(abstraction_junction)
    controlled_release.connect(abstraction_junction)
    spill.connect(abstraction_junction)

    abstraction_junction.connect(nonpublic_demand)
    abstraction_junction.connect(public_demand)
    abstraction_junction.connect(rivergauge)

    rivergauge.connect(outflow)

    reservoir_recorder = NumpyArrayStorageRecorder(
        model,
        storage_reservoir,
        name="reservoir_volume"
    )

    demand_recorder = NumpyArrayParameterRecorder(
        model,
        public_demand_max_flow,
        name="public_demand"
    )

    supply_recorder = NumpyArrayNodeRecorder(
        model,
        public_demand,
        name="public_supply"
    )

    rivergauge_recorder = NumpyArrayNodeRecorder(
        model,
        rivergauge,
        name="rivergauge_flow"
    )

    if controlled:
        index_recorder = NumpyArrayParameterRecorder(
            model, 
            control_curve_index, 
            name="cc_index"
        )

    model.run()

    results = reservoir_recorder.to_dataframe()
    results.columns = ["Reservoir Volume"]
    results.index = results.index.to_timestamp()

    supply_df = supply_recorder.to_dataframe()
    demand_df = demand_recorder.to_dataframe()
    rivergauge_df = rivergauge_recorder.to_dataframe()

    results["Public Supply"] = supply_df.values
    results["Public Demand"] = demand_df.values
    results["Percentage Full"] = results["Reservoir Volume"] / 463.45
    results["MRF Met"] = rivergauge_df.values >= 50.0

    if controlled:
        cc_index_df = index_recorder.to_dataframe()
        results["Control Curve Index"] = cc_index_df.values
    else:
        results["Control Curve Index"] = 0

    return results

def process_single_scenario(args):
    run, scenario, output_dir, controlled = args

    target_dir = output_dir / run.name
    os.makedirs(target_dir, exist_ok=True)

    shetran_run = run / scenario
    
    results = run_resource_model(shetran_run / "output_28001_discharge_sim_regulartimestep.txt", controlled=controlled)
    
    if controlled:
        filename = f"{scenario}_cc.csv"
    else:
        filename = f"{scenario}.csv"

    results.to_csv(target_dir / filename)

def run_resource_model_on_shetran_ensemble(ensemble_dir: Path):
    output_dir = Path("outputs/resource_model")

    scenarios = [
                "Baseline",
                "RCP2.6_10th",
                "RCP2.6_50th",
                "RCP2.6_90th",
                "RCP4.5_10th",
                "RCP4.5_50th",
                "RCP4.5_90th",
                "RCP6.0_10th",
                "RCP6.0_50th",
                "RCP6.0_90th",
                "RCP8.5_10th",
                "RCP8.5_50th",
                "RCP8.5_90th"
    ]

    runs = list(ensemble_dir.iterdir())

    tasks = [
        (run, scenario, output_dir, controlled) 
        for run in runs 
        for scenario in scenarios
        for controlled in [False, True]
    ]

    process_map(process_single_scenario, tasks, max_workers=32, chunksize=1)