import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from enduse.loadshape import (
    read_load_profiles_from_csvs,
    LoadShapesFromLoadProfiles,
)

resstock_path = str(
    (Path(__file__).parents[1] / "outputs/nrel_load_profiles/resstock/").as_posix()
)

comstock_path = str(
    (Path(__file__).parents[1] / "outputs/nrel_load_profiles/comstock/").as_posix()
)


# offset to PST
pst_offset = -pd.Timedelta(hours=3, minutes=15)

agg_cols = [
    {
        "label": "agg.electric.heat_pump",
        "agg_func": "sum",
        "agg_cols": [
            "out.electricity.cooling.energy_consumption",
            "out.electricity.heating.energy_consumption",
            "out.electricity.heating_supplement.energy_consumption",
        ],
    }
]

# pulling from dataframes
resstock = read_load_profiles_from_csvs(resstock_path)
resstock_loadshapes = LoadShapesFromLoadProfiles(
    load_profiles=resstock, timestamp_offset=pst_offset, agg_cols=agg_cols
)

# pulling from dir of .csvs
comstock_loadshapes = LoadShapesFromLoadProfiles(dir=comstock_path, agg_cols=agg_cols)

# netcdf output path
netcdf_path = str((Path(__file__).parents[1] / "outputs/loadshapes/").as_posix())
resstock_loadshapes.to_netcdf(netcdf_path + "/resstock_loadshapes.nc")
comstock_loadshapes.to_netcdf(netcdf_path + "/comstock_loadshapes.nc")

subset_params = {
    "in.geometry_building_type_recs": "Mobile Home",
    "in.puma": "G53011606",
}

with xr.open_dataset(netcdf_path + "/resstock_loadshapes.nc") as ds:
    res_subset = ds.sel(subset_params)
