import numpy as np
import pandas as pd
import xarray as xr
import dask
import time
from calendar import calendar
from datetime import timedelta
from datetime import datetime
import datetime as dt
from tzlocal import get_localzone
import pytz
from matplotlib.pyplot import axis, get
from enduse.oedi_tools import LoadProfiles
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from enduse.oedi_tools import LoadProfiles, rs_building_types, com_building_types
from sklearn import preprocessing
from typing import List, Tuple, Optional


def prepare_load_profiles_for_fit(loadprofiles: LoadProfiles):

    minute_offset = timedelta(minutes=15)

    fit_profile_list = {}

    for i in loadprofiles.load_profiles:

        # changing to pst
        loadprofiles.load_profiles[i]["timestamp"] = (
            loadprofiles.load_profiles[i]["timestamp"] - minute_offset
        )
        loadprofiles.load_profiles[i]["timestamp"] = pd.to_datetime(
            loadprofiles.load_profiles[i]["timestamp"]
        )

        # aggragteing to hourly
        loadprofiles.load_profiles[i] = loadprofiles.load_profiles[i].set_index(
            "timestamp"
        )

        # TODO make optional timezone change
        loadprofiles.load_profiles[i] = loadprofiles.load_profiles[i].tz_localize("EST")
        loadprofiles.load_profiles[i] = loadprofiles.load_profiles[i].tz_convert(
            get_localzone()
        )

        loadprofiles.load_profiles[i] = (
            loadprofiles.load_profiles[i].resample("H").sum()
        )

        # fixing the temperature column
        loadprofiles.load_profiles[i]["temperature"] = (
            loadprofiles.load_profiles[i]["temperature"] / 4
        )
        # converting celsius to fahrenheit
        loadprofiles.load_profiles[i]["temperature"] = (
            loadprofiles.load_profiles[i]["temperature"] * (9 / 5) + 32
        )

        # creating datetime related variables
        loadprofiles.load_profiles[i].reset_index(inplace=True)
        loadprofiles.load_profiles[i]["hour"] = loadprofiles.load_profiles[i][
            "timestamp"
        ].dt.hour
        loadprofiles.load_profiles[i]["dayofweek"] = loadprofiles.load_profiles[i][
            "timestamp"
        ].dt.dayofweek

        cal = calendar()
        holidays = cal.holidays(
            start=loadprofiles.load_profiles[i]["timestamp"].min(),
            end=loadprofiles.load_profiles[i]["timestamp"].max(),
        )

        loadprofiles.load_profiles[i]["holiday"] = loadprofiles.load_profiles[i][
            "timestamp"
        ].dt.date.isin(holidays.date)

        # creating seasonal variable
        # use time delta to inform cosine shift
        # TODO make timezonediff optional
        tz1 = pytz.timezone("US/Eastern")
        timezonediff = datetime.now().hour - datetime.now(tz1).hour
        index = np.linspace(0 + timezonediff, 8760 + timezonediff, num=8760)
        cosine = np.cos(2 * np.pi * index / 8760)

        loadprofiles.load_profiles[i]["cos"] = cosine

        # normalizing the load shapes
        load_profiles = loadprofiles.load_profiles[i].filter(
            like="out.electricity.", axis=1
        )
        min_max_scalar = preprocessing.MinMaxScaler()
        enduse_loadshape_norm = min_max_scalar.fit_transform(load_profiles)
        enduse_loadshape_norm = pd.DataFrame(enduse_loadshape_norm)

        enduse_loadshape_norm.columns = load_profiles.columns

        cols = ["timestamp", "temperature", "hour", "dayofweek", "holiday", "cos"]
        prepared_data = loadprofiles.load_profiles[i][cols]

        prepared_data = pd.concat([prepared_data, enduse_loadshape_norm], axis=1)

        fit_profile_list[i] = prepared_data

    return fit_profile_list


def fit_prepared_load_profiles(loadprofiles: dict):
    # fit vars
    colnames = ["temperature", "cos", "hour", "dayofweek", "holiday"]
    categorical_cols = ["hour", "dayofweek", "holiday"]
    profile_fit_list = {}

    for i in loadprofiles:

        vars = loadprofiles[i][colnames]
        for n in categorical_cols:
            vars[n] = pd.Categorical(vars[n])

        vars = pd.get_dummies(vars, prefix=categorical_cols)

        targets = loadprofiles[i].filter(like="out.electricity.", axis=1)

        targets.columns = targets.columns.str.replace("out.electricity.", "")
        targets.columns = targets.columns.str.replace(".energy_consumption", "")

        fit_list = {}

        for col in targets:
            target = targets[col]
            fit = HistGradientBoostingRegressor().fit(vars, target)
            fit_list[col + ".fit"] = fit

        profile_fit_list[i] = fit_list

    return profile_fit_list


def fit_load_profiles(load_profile_parms):

    loadprofiles = LoadProfiles(**load_profile_parms)
    loadprofiles = prepare_load_profiles_for_fit(loadprofiles)
    loadprofilefits = fit_prepared_load_profiles(loadprofiles)

    return loadprofilefits


class simple_HistGradientBoostingRegressor_fit:
    def __init__(
        self,
        segment: str,
        weather_type: str,
        state: str,
        puma_code: str,
        bldg_types: List[str],
    ):
        self.segment = segment
        self.weather_type = weather_type
        self.state = state
        self.puma_code = puma_code
        self.bldg_types = bldg_types

        load_profile_parms = {
            "segment": segment,
            "weather_type": weather_type,
            "state": state,
            "puma_code": puma_code,
            "bldg_types": bldg_types,
            "attach_temp": True,
        }
        self.hgbr_fit = fit_load_profiles(load_profile_parms)
