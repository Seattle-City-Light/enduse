import pandas as pd
import numpy as np
import warnings
import dask

from typing import List, Tuple, Optional
from pydantic import BaseModel, StrictStr, validator

state_abb = [
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

rs_building_types = [
    "mobile_home",
    "multi-family_with_2_-_4_units",
    "multi-family_with_5plus_units",
    "single-family_attached",
    "single-family_detached",
]

com_building_types = [
    "fullservicerestaurant",
    "retailstripmall",
    "warehouse",
    "retailstandalone",
    "smalloffice",
    "primaryschool",
    "mediumoffice",
    "secondaryschool",
    "outpatient",
    "quickservicerestaurant",
    "largeoffice",
    "largehotel",
    "smallhotel",
    "hospital",
]


def _generate_url_from_path_params(
    segment: str, weather_type: str, state: str, puma_code: str, bldg_type: str
) -> str:
    url = (
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
        + segment
        + "_"
        + weather_type
        + "_release_1/timeseries_aggregates/by_puma/state="
        + state
        + "/"
        + puma_code
        + "-"
        + bldg_type
        + ".csv"
    )
    return url


def _generate_meta_file_url_from_path_params(segment: str, weather_type: str) -> str:
    meta_file_url = (
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
        # + segment !!!! Not using provided segment since Commercial stock meta data doesn't have puma code mapped to county
        #               Once NREL fixes this issue change it to go to the commercial data set for the meta data
        + "resstock"
        + "_"
        + weather_type
        + "_release_1/metadata/metadata.parquet"
    )
    return meta_file_url


def _pull_county_code_for_puma(
    segment: str, weather_type: str, puma_code: str
) -> pd.DataFrame:
    meta_file_url = _generate_meta_file_url_from_path_params(segment, weather_type)

    try:
        meta_dat = pd.read_parquet(meta_file_url, columns=["in.puma", "in.county"])
    except:
        pass
        warnings.warn(f"{meta_file_url} meta file does not exist.")
        meta_dat = None

    if meta_dat is not None:
        meta_dat = meta_dat[meta_dat["in.puma"] == puma_code.upper()]

        meta_dat = meta_dat["in.county"].unique()
        if len(meta_dat) > 1:
            warnings.warn(
                f"Puma Code {puma_code} has muiltple County Codes associated with it in the meta file. Temperatures might be inaccurate. Using county code {meta_dat[0]}."
            )

        meta_dat = meta_dat[0]

    return meta_dat


def _generate_temp_file_url_from_path_params(
    segment: str, weather_type: str, puma_code: str
) -> str:
    meta_dat = _pull_county_code_for_puma(segment, weather_type, puma_code)

    temp_url = (
        "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
        + segment
        + "_"
        + weather_type
        + "_"
        + "release_1/weather/"
        + weather_type
        + "/"
        + meta_dat
        + "_"
        + weather_type
        + ".csv"
    )
    return temp_url


def _pull_puma_temp(segment: str, weather_type: str, puma_code: str) -> pd.Series:

    temp_url = _generate_temp_file_url_from_path_params(
        segment, weather_type, puma_code
    )

    try:
        temp_dat = pd.read_csv(temp_url)

    except:
        pass
        warnings.warn(f"{temp_url} temperature file does not exist.")
        temp_dat = None

    if temp_dat is not None:
        # temperature column has special character so using iloc
        temp_dat = temp_dat.iloc[:, 1]
        temp_dat = pd.Series(np.repeat(temp_dat.values, 4, axis=0))

    return temp_dat


def _pull_nrel_load_profiles(label: str, url: str) -> Tuple[str, pd.DataFrame]:
    """
    Function that makes the request to NREL data lake to download aggregate load profile .csv
    """
    try:
        bldg = pd.read_csv(url, infer_datetime_format=True, parse_dates=["timestamp"])
    except:
        pass
        warnings.warn(f"{url} csv file does not exist.")
        bldg = None
    return (label, bldg)


class LoadProfilePathValidator(BaseModel):
    segment: StrictStr
    weather_type: StrictStr
    state: StrictStr
    puma_code: StrictStr
    bldg_type: StrictStr

    @validator("segment")
    def check_valid_segment(cls, v):
        if v not in ["resstock", "comstock"]:
            raise ValueError(f"{v} is not a valid segment type (resstock or comstock)")
        return v

    @validator("weather_type")
    def check_valid_weather_type(cls, v):
        if v not in ["tmy3", "amy2018"]:
            raise ValueError(f"{v} is not a valid weather type (tmy3 or amy2018)")
        return v

    @validator("bldg_type")
    def check_valid_building_type(cls, v, values):
        if "segment" in values.keys() and v is not None:
            if values["segment"] == "resstock" and v not in rs_building_types:
                raise ValueError(
                    f"{v} is not a valid building type for {values['segment']}. Please select a vaild building type from this list: \n {rs_building_types}"
                )

            if values["segment"] == "comstock" and v not in com_building_types:
                raise ValueError(
                    f"{v} is not a valid building type for {values['segment']}. Please select a vaild building type from this list: \n {com_building_types}"
                )

        return v

    @validator("state")
    def check_valid_state_abb(cls, v):
        if v not in state_abb:
            raise ValueError(
                f"{v} is not a valid state abbreviation (NY, WA, CA, etc.)"
            )
        return v

    @validator("puma_code")
    def check_valid_puma_code(cls, v):
        if len(v) != 9 or v[0] != "g":
            raise ValueError(
                f"{v} is not a valid puma code (check format i.g. g53011606)"
            )
        return v


class LoadProfiles:
    def __init__(
        self,
        segment: str,
        weather_type: str,
        state: str,
        puma_code: str,
        bldg_types: List[str],
        validate_paths: bool = True,
        pull_load_profiles: bool = True,
        use_dask: bool = True,
        attach_temp: bool = False,
    ):
        self.segment = segment
        self.weather_type = weather_type
        self.state = state
        self.puma_code = puma_code
        self.bldg_types = bldg_types
        self.validate_paths = validate_paths
        self.attach_temp = attach_temp

        # url validation is optional
        if validate_paths:
            self._validate_load_profile_paths()

        self._urls = self._generate_urls()

        # pulling load profiles on instantion is optional
        if pull_load_profiles:
            # default behavior is to use dask to pull in parallel
            self._load_profiles = self.pull_load_profiles(use_dask)
        else:
            self._load_profiles - None

    @property
    def urls(self):
        return self._urls

    @property
    def load_profiles(self):
        return self._load_profiles

    def _validate_load_profile_paths(self) -> None:
        for i in self.bldg_types:
            LoadProfilePathValidator(
                segment=self.segment,
                weather_type=self.weather_type,
                state=self.state,
                puma_code=self.puma_code,
                bldg_type=i,
            )

    def _generate_urls(self) -> List[str]:
        urls = []
        for i in self.bldg_types:
            urls.append(
                _generate_url_from_path_params(
                    self.segment, self.weather_type, self.state, self.puma_code, i
                )
            )
        return urls

    # not a protected method since user may want to review urls manually before pulling .csv files
    def pull_load_profiles(self, use_dask: bool = True) -> None:
        load_profiles = []

        if use_dask:
            for i, x in zip(self.bldg_types, self.urls):
                load_profiles.append(dask.delayed(_pull_nrel_load_profiles)(i, x))
            load_profiles = dask.compute(*load_profiles)
        else:
            for i in self.urls:
                load_profiles.append(_pull_nrel_load_profiles(i, x))

        if self.attach_temp:
            puma_temps = _pull_puma_temp(
                self.segment, self.weather_type, self.puma_code
            )

            for i in load_profiles:
                if i[1] is not None:
                    i[1]["temperature"] = puma_temps

        return dict(load_profiles)

    def load_profiles_to_csv(
        self, path: str, use_dask: bool = True, file_names: Optional[str] = None
    ) -> None:
        base_path = (
            f"{path}/{self.segment}_{self.weather_type}_{self.state}_{self.puma_code}_"
        )

        outputs = []

        if use_dask:
            for i, x in self.load_profiles.items():
                # avoid trying to export None types for invalid urls
                if x is not None:
                    path = base_path + i + ".csv"
                    outputs.append(dask.delayed(x.to_csv)(path, index=False))
            dask.compute(*outputs)
        else:
            for i, x in self.load_profiles.items():
                if x is not None:
                    path = base_path + i + ".csv"
                    x.to_csv(path, index=False)
