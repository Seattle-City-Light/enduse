import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from glob import glob

from enduse.loadshape import (
    read_load_profiles_from_csvs,
    LoadShapesFromLoadProfiles,
)

resstock_path = str(
    (Path(__file__).parents[2] / "outputs/nrel_load_profiles/resstock/").as_posix()
)

comstock_path = str(
    (Path(__file__).parents[2] / "outputs/nrel_load_profiles/comstock/").as_posix()
)


class TestLoadProfilesFromFiles:

    csvs = glob(f"{resstock_path}/*.csv")
    load_profiles = read_load_profiles_from_csvs(resstock_path)

    def test_load_profiles_from_csvs_keys(self):
        assert sorted(list(self.load_profiles.keys())) == sorted(self.csvs)

    def test_load_profiles_from_csvs_values(self):
        assert all([type(i) == pd.DataFrame for i in self.load_profiles.values()])


# TODO build out other tests for load_profiles
