from attr import validate
import pandas as pd
import warnings
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


def pull_nrel_load_shapes(
    segment: str, weather_type: str, bldg_type: str, state: str, puma_code: str,
) -> pd.DataFrame:

    # URL to the NREL data lake csv
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

    # checking if file exists
    try:
        bldg = pd.read_csv(url, infer_datetime_format=True, parse_dates=["timestamp"])

    except:
        pass
        warnings.warn(f"{url} csv file does not exist.")
        bldg = None

    return bldg


class PullLoadShape(BaseModel):
    segment: StrictStr
    weather_type: StrictStr
    bldg_type: StrictStr
    state: StrictStr
    puma_code: StrictStr

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "load_shapes", self.get_load_shapes())

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

        if "segment" in values.keys():

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

    def get_load_shapes(self):
        load_shapes = pull_nrel_load_shapes(
            self.segment, self.weather_type, self.bldg_type, self.state, self.puma_code
        )

        return load_shapes

