# need to create Equipment and RampEfficiency objects to test ramp
import numpy as np
import pandas as pd
import xarray as xr
import dask
from calendar import calendar
from datetime import timedelta
from matplotlib.pyplot import axis
from enduse.oedi_tools import LoadProfiles
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import TimeSeriesSplit


def simple_HistGradientBoostingRegressor_fit(
    loadprofiles_path: str,
) -> HistGradientBoostingRegressor:

    loadprofiles_path = "I:/FINANCE/FPU/Matt/END USE/enduse/outputs/nrel_load_profiles/resstock/resstock_tmy3_WA_g53011606_mobile_home.csv"

    loadprofiles = pd.read_csv(loadprofiles_path, parse_dates=["timestamp"])
    meta = pd.read_parquet(
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_tmy3_release_1/metadata/metadata.parquet",
        columns=["in.puma", "in.county"],
    )

    puma_code = loadprofiles.iloc[0]["in.puma"]

    meta = meta[meta["in.puma"] == puma_code]
    county_code = meta.iloc[0]["in.county"]

    temps = pd.read_csv(
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_tmy3_release_1/weather/tmy3/"
        + county_code
        + "_tmy3.csv",
        parse_dates=["date_time"],
    )

    temps.to_csv("C:/Users/HamlinM/Desktop/g53011606temps.csv")
