import itertools
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


# residential agg_cols
resstock_df = pd.read_csv(
    resstock_path + "/resstock_tmy3_WA_g53011601_single-family_detached.csv"
)

res_agg_cols = {}
res_agg_cols["agg.electric.heating.energy_consumption"] = [
    "out.electricity.heating.energy_consumption",
    "out.electricity.heating_supplement.energy_consumption",
]

res_agg_cols["agg.electric.cooling.energy_consumption"] = [
    "out.electricity.cooling.energy_consumption"
]

res_agg_cols["agg.electric.heat_pump.energy_consumption"] = (
    res_agg_cols["agg.electric.heating.energy_consumption"]
    + res_agg_cols["agg.electric.cooling.energy_consumption"]
)

res_agg_cols["agg.electric.laundry.energy_consumption"] = [
    "out.electricity.clothes_dryer.energy_consumption",
    "out.electricity.clothes_washer.energy_consumption",
]

res_agg_cols["agg.electric.cooking.energy_consumption"] = [
    "out.electricity.cooking_range.energy_consumption",
    "out.electricity.dishwasher.energy_consumption",
    "out.electricity.freezer.energy_consumption",
    "out.electricity.refrigerator.energy_consumption",
]

res_agg_cols["agg.electric.lighting.energy_consumption"] = [
    "out.electricity.ext_holiday_light.energy_consumption",
    "out.electricity.exterior_lighting.energy_consumption",
    "out.electricity.garage_lighting.energy_consumption",
    "out.electricity.interior_lighting.energy_consumption",
]

res_agg_cols["agg.electric.water_heating.energy_consumption"] = [
    "out.electricity.water_systems.energy_consumption",
    "out.electricity.vehicle.energy_consumption",
]

res_exclude_cols = [
    "out.electricity.pv.energy_consumption",
    "out.electricity.vehicle.energy_consumption",
]

res_elec_cols = (
    resstock_df.loc[:, (resstock_df != 0).any(axis=0)]
    .filter(like="out.electricity")
    .columns.tolist()
)

# get all other end_uses
res_agg_cols["agg.electric.other.energy_consumption"] = list(
    set(res_elec_cols)
    - set(res_exclude_cols)
    - set(itertools.chain(*list(res_agg_cols.values())))
)

res_agg = []
for i, x in res_agg_cols.items():
    res_agg.append(
        {"label": i, "agg_func": "sum", "agg_cols": x,}
    )

# pulling from dataframes
resstock = read_load_profiles_from_csvs(resstock_path)
resstock_loadshapes = LoadShapesFromLoadProfiles(
    load_profiles=resstock, timestamp_offset=pst_offset, agg_cols=res_agg
)

# pulling from dir of .csvs
comstock_loadshapes = LoadShapesFromLoadProfiles(dir=comstock_path)

# netcdf output path
netcdf_path = str((Path(__file__).parents[1] / "outputs/loadshapes/").as_posix())
resstock_loadshapes.to_netcdf(netcdf_path + "/resstock_loadshapes.nc")
comstock_loadshapes.to_netcdf(netcdf_path + "/comstock_loadshapes.nc")

subset_params = {
    "shape.type": "Load Shape",
    "in.geometry_building_type_recs": "Mobile Home",
    "in.puma": "G53011606",
}

with xr.open_dataset(netcdf_path + "/resstock_loadshapes.nc") as ds:
    res_subset = ds.sel(subset_params)
