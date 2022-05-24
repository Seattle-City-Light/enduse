from copy import deepcopy
import numpy as np
import pandas as pd

from pathlib import Path

from enduse.stockobjects import Building

from parse_csv_utils import create_end_uses
from enduse.stockturnover import BuildingModel


# run model here
# set load shape path
csv_path = str((Path(__file__).parents[1] / "examples/inputs/residential/").as_posix())
netcdf_path = str(
    (
        Path(__file__).parents[1] / "examples/inputs/residential/resstock_loadshapes.nc"
    ).as_posix()
)

saturation = pd.read_csv(csv_path + "/saturations.csv")
fuel_share = pd.read_csv(csv_path + "/fuelSharesByVintage - Seattle Code Update.csv")
efficiency_share = pd.read_csv(csv_path + "/efficiencyShares.csv")
measure_inputs = pd.read_csv(csv_path + "/equipmentMeasureInputs.csv")
standards = pd.read_csv(csv_path + "/equipmentStandards.csv")

# testing 2 different load shape files
load_shapes_1 = pd.read_csv(csv_path + "/loadShapes_G53011601.csv")
load_shapes_2 = pd.read_csv(csv_path + "/loadShapes_G53011606.csv")

# drop electric vehicles from measures
measure_inputs = measure_inputs[~measure_inputs["End Use"].isin(["Electric Vehicles"])]

# standards start in 2022 need to allign with forecast range which starts in 2021
standards["2021"] = standards["2022"]

standards_mapped = (
    measure_inputs[["Efficiency Description", "Efficiency Level"]]
    .drop_duplicates()
    .set_index("Efficiency Description")
    .to_dict()["Efficiency Level"]
)

# single family tests
sf_end_use_params_1 = {
    "segment": "Single Family",
    "vintage": "Existing",
    "efficiency_share": efficiency_share,
    "measure_inputs": measure_inputs,
    "standards": standards,
    "saturation": saturation,
    "load_shape": load_shapes_1,
    "fuel_share": fuel_share,
    "start_year": 2021,
    "end_year": 2041,
    "load_shape_path": netcdf_path,
}

sf_end_use_params_2 = deepcopy(sf_end_use_params_1)
sf_end_use_params_2.update({"load_shape": load_shapes_2})

sf_end_uses_1 = create_end_uses(**sf_end_use_params_1)
sf_end_uses_2 = create_end_uses(**sf_end_use_params_2)

sf_building_1 = {
    "building_label": "Single Family G53011601",
    "end_uses": sf_end_uses_1,
    "start_year": 2021,
    "end_year": 2041,
    "building_stock": np.linspace(206570, 206570, 21).tolist(),
    "segment": "Single Family",
    "construction_vintage": "Existing",
}

sf_building_2 = deepcopy(sf_building_1)
sf_building_2.update(
    {"building_label": "Single Family G53011606", "end_uses": sf_end_uses_2}
)

sf_building_parsed_1 = Building(**sf_building_1)
sf_stockturnover_1 = BuildingModel(sf_building_parsed_1)

sf_building_parsed_2 = Building(**sf_building_2)
sf_stockturnover_2 = BuildingModel(sf_building_parsed_2)

# multi family tests
mf_end_use_params_1 = {
    "segment": "Multifamily - High Rise",
    "vintage": "Existing",
    "efficiency_share": efficiency_share,
    "measure_inputs": measure_inputs,
    "standards": standards,
    "saturation": saturation,
    "load_shape": load_shapes_1,
    "fuel_share": fuel_share,
    "start_year": 2021,
    "end_year": 2041,
    "load_shape_path": netcdf_path,
}

mf_end_uses_1 = create_end_uses(**mf_end_use_params_1)

mf_building_1 = {
    "building_label": "Multifamily - High Rise G53011601",
    "end_uses": mf_end_uses_1,
    "start_year": 2021,
    "end_year": 2041,
    "building_stock": np.linspace(64650, 64650, 21).tolist(),
    "segment": "Multifamily - High Rise",
    "construction_vintage": "Existing",
}

mf_building_parsed_1 = Building(**mf_building_1)
mf_stockturnover_1 = BuildingModel(mf_building_parsed_1)

sf1_equip_count = sf_stockturnover_1.summarize_by(
    coord="end_use_label", data_dim="equipment_stock", agg_func="sum"
)

sf2_load_shape = sf_stockturnover_2.summarize_by(
    coord="load_shape", data_dim="consumption_shaped", agg_func="sum"
)

mf_load_shape = mf_stockturnover_1.summarize_by(
    coord="load_shape", data_dim="consumption_shaped", agg_func="sum"
)
