import calendar

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd

from pathlib import Path
from shapely.geometry import mapping

def calculate_baseline_temps() -> pd.DataFrame:
    data_dir = Path("data/chess/met/tas")

    BASELINE_START_YEAR = 1980
    BASELINE_END_YEAR = 2000

    K_convert = -273.15

    gdf = gpd.read_file(Path("data/nfra/nrfa_28001.zip"))

    df_list = []
    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR):
        for month in range(1,13):
            _, last_day = calendar.monthrange(year, month)
            ds_path = data_dir / f"chess-met_tas_gb_1km_daily_{year}{month:02d}01-{year}{month:02d}{last_day}.nc"

            ds = xr.open_dataset(ds_path)
            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            ds.rio.write_crs("EPSG:27700", inplace=True)

            data_to_clip = ds['tas']
            
            clipped_ds = data_to_clip.rio.clip(
                gdf.geometry.apply(mapping),
                gdf.crs,
                drop=True,
                all_touched=True
            )
            
            avg_timeseries = clipped_ds.mean(dim=["x", "y"])

            df_list.append(avg_timeseries.to_dataframe())

    df = pd.concat(df_list)

    df = df.drop(columns=["spatial_ref", "crsOSGB"])

    df["tas"] = df["tas"] + K_convert

    df = df.groupby([df.index.month, df.index.day]).mean()

    df.index.names = ["month", "day"]

    return df

def calculate_baseline_pet() -> pd.DataFrame:
    data_dir = Path("data/chess/pe")

    BASELINE_START_YEAR = 1980
    BASELINE_END_YEAR = 2000

    K_convert = -273.15

    gdf = gpd.read_file(Path("data/nfra/nrfa_28001.zip"))

    df_list = []
    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR):
        for month in range(1,13):
            _, last_day = calendar.monthrange(year, month)
            ds_path = data_dir / f"chess-pe_pet_gb_1km_daily_{year}{month:02d}01-{year}{month:02d}{last_day}.nc"

            ds = xr.open_dataset(ds_path)
            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            ds.rio.write_crs("EPSG:27700", inplace=True)

            data_to_clip = ds['pet']
            
            clipped_ds = data_to_clip.rio.clip(
                gdf.geometry.apply(mapping),
                gdf.crs,
                drop=True,
                all_touched=True
            )
            
            avg_timeseries = clipped_ds.mean(dim=["x", "y"])

            df_list.append(avg_timeseries.to_dataframe())

    df = pd.concat(df_list)

    df = df.drop(columns=["spatial_ref", "crsOSGB"])

    df = df.groupby([df.index.month, df.index.day]).mean()

    df.index.names = ["month", "day"]

    return df

def get_temp_anomoly(year:int, scenario:str, percentile:str) -> list[np.float32]:
    scenario_1 = scenario[3]
    scenario_2 = scenario[5]
    
    data_path = Path(f"data/ukcp18/{scenario}/tasAnom_rcp{scenario_1}{scenario_2}_land-prob_uk_25km_cdf_b8100_1y_seas_{year-1}1201-{year}1130.nc")
    ds = xr.open_dataset(data_path)    
    
    target_x = 412500
    target_y = 387500

    specific_cell_data = ds.sel(
        projection_x_coordinate=target_x, 
        projection_y_coordinate=target_y, 
        method="nearest"
    )

    df = specific_cell_data["tasAnom"].to_pandas()

    season_to_month = {0: 7, 1: 10, 2: 1, 3: 4}

    uplifts = []
    for season_idx in range(4):
        target_month = season_to_month[season_idx]

        uplift = df.loc[df.index.month == target_month, float(percentile)].values[0]

        uplifts.append(uplift)

    return uplifts

def create_synthetic_temps(baseline_temps:pd.DataFrame) -> pd.DataFrame:
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

    month_to_season = {
        6: 0, 7: 0, 8: 0,  # Summer
        9: 1, 10: 1, 11: 1, # Autumn
        12: 2, 1: 2, 2: 2,  # Winter
        3: 3, 4: 3, 5: 3    # Spring
    }

    timeseries["month"] = timeseries.index.month
    timeseries["day"] = timeseries.index.day

    timeseries = timeseries.join(baseline_temps["tas"], on=["month", "day"])

    timeseries["Baseline"] = timeseries["tas"]

    for column_name in timeseries.columns:
        if column_name in ["Baseline", "month", "day", "tas"]:
            pass
        else:
            scenario = column_name[:6]
            percentile = column_name[-4:-2]

            for year in range(START_YEAR, END_YEAR+1):
                new_start = f"{year-1}-12-01"
                new_end = f"{year}-11-30"

                uplifts = get_temp_anomoly(year, scenario, percentile)

                month_uplift_map = {
                    month: uplifts[season_idx] 
                    for month, season_idx in month_to_season.items()
                }

                year_slice = slice(new_start, new_end)

                current_months = timeseries.loc[year_slice, "month"]

                timeseries.loc[year_slice, column_name] = (
                    timeseries.loc[year_slice, "Baseline"] + current_months.map(month_uplift_map)
                )

    timeseries = timeseries.drop(columns=["month", "day", "tas"])

    timeseries = timeseries.astype(float)

    output_dir = Path("outputs/shetran_inputs")
    timeseries.to_csv(output_dir / "temps.csv")

    return timeseries

def fit_pet_model() -> tuple[pd.DataFrame, np.poly1d]:
    temp_data_dir = Path("data/chess/met/tas")
    pet_data_dir = Path("data/chess/pe")

    BASELINE_START_YEAR = 1961
    BASELINE_END_YEAR = 2019

    K_convert = -273.15

    gdf = gpd.read_file(Path("data/nfra/nrfa_28001.zip"))

    temp_df_list = []
    pet_df_list = []
    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR):
        for month in range(1,13):
            _, last_day = calendar.monthrange(year, month)

            temp_ds_path = temp_data_dir / f"chess-met_tas_gb_1km_daily_{year}{month:02d}01-{year}{month:02d}{last_day}.nc"

            temp_ds = xr.open_dataset(temp_ds_path)
            temp_ds = temp_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            temp_ds.rio.write_crs("EPSG:27700", inplace=True)

            data_to_clip = temp_ds['tas']
            
            clipped_temp_ds = data_to_clip.rio.clip(
                gdf.geometry.apply(mapping),
                gdf.crs,
                drop=True,
                all_touched=True
            )
            
            avg_temp_timeseries = clipped_temp_ds.mean(dim=["x", "y"])

            temp_df_list.append(avg_temp_timeseries.to_dataframe())

            pet_ds_path = pet_data_dir / f"chess-pe_pet_gb_1km_daily_{year}{month:02d}01-{year}{month:02d}{last_day}.nc"

            pet_ds = xr.open_dataset(pet_ds_path)
            pet_ds = pet_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            pet_ds.rio.write_crs("EPSG:27700", inplace=True)

            data_to_clip = pet_ds['pet']
            
            clipped_pet_ds = data_to_clip.rio.clip(
                gdf.geometry.apply(mapping),
                gdf.crs,
                drop=True,
                all_touched=True
            )
            
            avg_pet_timeseries = clipped_pet_ds.mean(dim=["x", "y"])

            pet_df_list.append(avg_pet_timeseries.to_dataframe())

    full_temp = pd.concat(temp_df_list)
    full_pet = pd.concat(pet_df_list)

    cols_to_drop = ["spatial_ref", "crsOSGB"]
    full_temp = full_temp.drop(columns=cols_to_drop, errors="ignore")
    full_pet = full_pet.drop(columns=cols_to_drop, errors="ignore")

    df = full_temp.join(full_pet, how="inner")

    df["tas"] = df["tas"] + K_convert

    X = df["tas"].values
    y = df["pet"].values

    coeffs = np.polyfit(X, y, 2)
    model_func = np.poly1d(coeffs)

    df["modelPet"] = model_func(df["tas"]).clip(min=0)
        
    return df, model_func

def create_synthetic_pet(baseline_pet: pd.DataFrame, synthetic_temps: pd.DataFrame, pet_model: np.poly1d) -> pd.DataFrame:
    mapper = pd.DataFrame(index=synthetic_temps.index)
    mapper["month"] = mapper.index.month
    mapper["day"] = mapper.index.day
    
    observed_baseline = mapper.reset_index().merge(
        baseline_pet[["pet"]], 
        left_on=["month", "day"], 
        right_index=True, 
        how="left"
    ).set_index("index")["pet"]

    raw_values = pet_model(synthetic_temps)
    
    modeled_all = pd.DataFrame(
        raw_values, 
        index=synthetic_temps.index, 
        columns=synthetic_temps.columns
    )

    uplifts = modeled_all.subtract(modeled_all["Baseline"], axis=0)

    final_pet = uplifts.add(observed_baseline, axis=0).clip(lower=0)

    final_pet["Baseline"] = observed_baseline

    output_dir = Path("outputs/shetran_inputs")
    final_pet.astype(float).to_csv(output_dir / "pet.csv")

    return final_pet.astype(float)