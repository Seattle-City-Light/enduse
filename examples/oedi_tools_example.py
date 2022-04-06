from ast import Load
import numpy as np
import pandas as pd

from pathlib import Path
from copy import deepcopy

from enduse.oedi_tools import LoadProfiles, rs_building_types, com_building_types

profile_path = Path(__file__).parents[1]


# example pulling a single PUMA
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

# Example pulling multiple pumas
pumas = ["g53011601", "g53011602"]

print("Pulling multiple resstock pumas")

multi_res_profiles = []
for i in pumas:
    res_oedi_puma.update({"puma_code": i})
    res_profile = LoadProfiles(**res_oedi_puma)
    multi_res_profiles.append(res_profile)


print("Exporting multiple resstock pumas")
for i in multi_res_profiles:
    i.load_profiles_to_csv(
        str((profile_path / "outputs/nrel_load_profiles/resstock/").as_posix())
    )


print("Pulling multiple comstock pumas")

multi_com_profiles = []
for i in pumas:
    com_oedi_puma.update({"puma_code": i})
    com_profile = LoadProfiles(**com_oedi_puma)
    multi_com_profiles.append(com_profile)


print("Exporting multiple comstock pumas")
for i in multi_com_profiles:
    i.load_profiles_to_csv(
        str((profile_path / "outputs/nrel_load_profiles/comstock/").as_posix())
    )
