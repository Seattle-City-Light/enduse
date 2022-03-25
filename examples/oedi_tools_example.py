import numpy as np
import pandas as pd

from pathlib import Path

from enduse.oedi_tools import LoadProfiles, rs_building_types, com_building_types

profile_path = Path(__file__).parents[1]

res_oedi_puma = {
    "segment": "resstock",
    "weather_type": "tmy3",
    "state": "WA",
    "puma_code": "g53011606",
    "bldg_types": rs_building_types,
}

print("Pulling resstock")

res_profiles = LoadProfiles(**res_oedi_puma)

print("Exporting resstock to .csv")

res_profiles.load_profiles_to_csv(
    str((profile_path / "outputs/nrel_load_profiles/resstock/").as_posix())
)

com_oedi_puma = {
    "segment": "comstock",
    "weather_type": "tmy3",
    "state": "WA",
    "puma_code": "g53011606",
    "bldg_types": com_building_types,
}

print("Pulling comstock")
com_profiles = LoadProfiles(**com_oedi_puma)

print("Exporting comstock to .csv")
com_profiles.load_profiles_to_csv(
    str((profile_path / "outputs/nrel_load_profiles/comstock/").as_posix())
)
