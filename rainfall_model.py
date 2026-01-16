import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from scipy.stats import gamma
from tqdm import tqdm
from time import time

def calculate_transition_matricies(df):
    # Season, From, To

    # Summer. Autumn, Winter, Spring
    # Dry, Wet, Extreme
    # Dry, Wet, Extreme
    
    trans_matrix = np.zeros((4, 3, 3))

    data = df.copy()

    month_to_season = {
        6: 0, 7: 0, 8: 0,  # Summer
        9: 1, 10: 1, 11: 1, # Autumn
        12: 2, 1: 2, 2: 2,  # Winter
        3: 3, 4: 3, 5: 3    # Spring
    }

    data["Season"] = data.index.month.map(month_to_season)

    thresholds = {}
    
    for s in range(4):
        thresholds[s] = data.loc[
            (data["Season"] == s) & (data["Rainfall (mm)"] > 0), 
            "Rainfall (mm)"
        ].quantile(0.95)

    data["Season_Threshold"] = data["Season"].map(thresholds)

    conditions = [
        data["Rainfall (mm)"] == 0,
        data["Rainfall (mm)"] >= data["Season_Threshold"]
    ]

    choices = [0, 2]

    data["State"] = np.select(conditions, choices, default=1)

    data["Next_State"] = data["State"].shift(-1)

    data = data.dropna(subset=["Next_State"])

    for season_idx in range(4):
        season_data = data[data["Season"] == season_idx]

        counts = pd.crosstab(season_data["State"], season_data["Next_State"])

        counts = counts.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)

        counts_arr = counts.values
        counts_arr = counts_arr.astype(float)
        row_sums = counts_arr.sum(axis=1, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            prob_matrix = np.divide(
                counts_arr, 
            row_sums, 
            out=np.zeros_like(counts_arr), 
            where=row_sums!=0
        )

        trans_matrix[season_idx] = np.nan_to_num(prob_matrix)

    return trans_matrix

def fit_gamma_function(df):
    # Season, Weather Type, Gamma Parameters

    # Summer. Autumn, Winter, Spring
    # Wet, Extreme
    # Alpha, Location, Beta
    
    parameter_matricies = np.zeros((4, 3))

    data = df.copy()

    month_to_season = {
        6: 0, 7: 0, 8: 0,  # Summer
        9: 1, 10: 1, 11: 1, # Autumn
        12: 2, 1: 2, 2: 2,  # Winter
        3: 3, 4: 3, 5: 3    # Spring
    }

    data["Season"] = data.index.month.map(month_to_season)

    thresholds = {}
    
    for s in range(4):
        thresholds[s] = data.loc[
            (data["Season"] == s) & (data["Rainfall (mm)"] > 0), 
            "Rainfall (mm)"
        ].quantile(0.95)

    data["Season_Threshold"] = data["Season"].map(thresholds)

    conditions = [
        data["Rainfall (mm)"] == 0,
        data["Rainfall (mm)"] >= data["Season_Threshold"]
    ]

    choices = [0, 2]

    data["State"] = np.select(conditions, choices, default=1)

    for season_idx in range(4):
        for state_idx in range(1,3):
            alpha, loc, beta = gamma.fit(data.loc[(data["Season"] == season_idx) & (data["Rainfall (mm)"] > 0), "Rainfall (mm)"], floc=0)
            parameter_matricies[season_idx, 0] = alpha
            parameter_matricies[season_idx, 1] = loc
            parameter_matricies[season_idx, 2] = beta

    return parameter_matricies

def markov_model(trans_mat, param_mat, start, end):
    dates = pd.date_range(start, end)
    n_days = len(dates)

    season_lookup = np.array([0, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2])

    months = dates.month.values
    seasons = season_lookup[months]

    random_numbers = np.random.rand(n_days)

    states = np.zeros(n_days, dtype=int)
    states[0] = np.random.randint(0, 3)

    for t in range(1, n_days):
        current_season = seasons[t]
        prev_state = states[t-1]

        probs = trans_mat[current_season, prev_state, :]
        cum_probs = np.cumsum(probs)
        state_idx = np.searchsorted(cum_probs, random_numbers[t])
        states[t] = state_idx

    rainfall = np.zeros(n_days)

    for season_idx in range(4):
        alpha = param_mat[season_idx, 0]
        loc   = param_mat[season_idx, 1]
        beta  = param_mat[season_idx, 2]

        mask_s1 = (seasons == season_idx) & (states == 1)
        n_s1 = np.sum(mask_s1)
        
        if n_s1 > 0:
            u_s1 = np.random.uniform(0.00, 0.95, size=n_s1)
            rainfall[mask_s1] = gamma.ppf(u_s1, a=alpha, loc=loc, scale=beta)

        mask_s2 = (seasons == season_idx) & (states == 2)
        n_s2 = np.sum(mask_s2)
        
        if n_s2 > 0:
            u_s2 = np.random.uniform(0.95, 0.99999, size=n_s2)
            rainfall[mask_s2] = gamma.ppf(u_s2, a=alpha, loc=loc, scale=beta)
    
    return rainfall

def get_precipitation_perturbations(parameter_matricies, year, scenario, percentile):
    parameter_matricies = parameter_matricies.copy()
    
    scenario_1 = scenario[3]
    scenario_2 = scenario[5]
    
    data_path = Path(f"data/ukcp18/{scenario}/prAnom_rcp{scenario_1}{scenario_2}_land-prob_uk_25km_cdf_b8100_1y_seas_{year-1}1201-{year}1130.nc")
    ds = xr.open_dataset(data_path)

    target_x = 412500
    target_y = 387500

    specific_cell_data = ds.sel(
        projection_x_coordinate=target_x, 
        projection_y_coordinate=target_y, 
        method="nearest"
    )

    df = specific_cell_data["prAnom"].to_pandas()

    season_to_month = {0: 7, 1: 10, 2: 1, 3: 4}

    for season_idx in range(4):
        target_month = season_to_month[season_idx]

        uplift = df.loc[df.index.month == target_month, float(percentile)].values[0]

        #print(f"Scenario: {scenario}, Percentile: {percentile}, Year: {year}, Uplift: {uplift}")
        
        current_beta = parameter_matricies[season_idx, 2]
        parameter_matricies[season_idx, 2] = current_beta * (1 + (uplift / 100))

    return parameter_matricies

def produce_timeseries(trans_mat, param_mat):
    START_YEAR = 2027
    END_YEAR = 2099

    timeseries_index = pd.date_range(start=f"{START_YEAR-1}-12-01", end=f"{END_YEAR}-11-30")

    timeseries = pd.DataFrame(
        columns=[
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
        ],
        index=timeseries_index
    )

    timeseries.index.name = "Date"

    for column_name in timeseries.columns:
        if column_name == "Baseline":
            timeseries["Baseline"] = markov_model(trans_mat, param_mat, f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
        else:
            scenario = column_name[:6]
            percentile = column_name[-4:-2]

            for year in range(START_YEAR, END_YEAR+1):
                new_start = f"{year-1}-12-01"
                new_end = f"{year}-11-30"

                new_param_mat = get_precipitation_perturbations(param_mat, year, scenario, percentile)
                timeseries.loc[new_start:new_end, column_name] = markov_model(trans_mat, new_param_mat, new_start, new_end)

    timeseries = timeseries.astype(float)
    
    return timeseries

def produce_timeseries_ensemble(transition_matricies, parameter_matricies, output_dir, n_runs):
    with tqdm(range(n_runs), unit="run") as pbar:
        for i in pbar:
            start_time = time()
            
            timeseries = produce_timeseries(transition_matricies, parameter_matricies)
            
            timeseries.to_csv(output_dir / f"run_{i+1}.csv")
            
            elapsed_seconds = time() - start_time
            minutes_per_run = elapsed_seconds / 60
            
            pbar.set_postfix({"min/run": f"{minutes_per_run:.2f}"})