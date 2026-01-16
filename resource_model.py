import pandas as pd

from pywr.model import Model
from pywr.nodes import Catchment, Link, Output, Storage, RiverGauge
from pywr.parameters import DataFrameParameter, MonthlyProfileParameter
from pywr.recorders import NumpyArrayStorageRecorder

from pathlib import Path

def run_resource_model(flows_path: Path):
    flows = pd.read_csv(flows_path)

    dates = pd.date_range("2026-12-01", "2099-11-30")
    flows.index = dates

    flows["discharge"] = flows["discharge at the outlet - regular timestep   24.00 hours"] * 3600 * 24 / 1000

    model = Model()

    model.timestepper.start = pd.Timestamp("2026-12-01")
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
        cost=controlled_release_cost_profile
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

    public_demand = Output(
        model,
        "public_demand",
        max_flow= public_demand_profile,
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

    storage_reservoir = Storage(
        model,
        "storage_reservoir",
        cost= 0,
        min_volume= 0.0,
        max_volume= 46345.0,
        initial_volume= 46345.0,
        position= (42.0903, 31.8102)
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

    recorder = NumpyArrayStorageRecorder(
        model,
        storage_reservoir
    )

    model.run()

    results = recorder.to_dataframe()

    results.columns = ["Reservoir Volume"]
    results.index = results.index.to_timestamp()

    results["Percentage Full"] = results["Reservoir Volume"] / 463.45

    return results