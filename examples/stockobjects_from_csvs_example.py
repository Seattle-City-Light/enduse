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
load_shapes = pd.read_csv(csv_path + "/loadShapes.csv")

# standards start in 2022 need to allign with forecast range which starts in 2021
standards["2021"] = standards["2022"]

standards_mapped = (
    measure_inputs[["Efficiency Description", "Efficiency Level"]]
    .drop_duplicates()
    .set_index("Efficiency Description")
    .to_dict()["Efficiency Level"]
)

end_use_params = {
    "segment": "Single Family",
    "vintage": "Existing",
    "efficiency_share": efficiency_share,
    "measure_inputs": measure_inputs,
    "standards": standards,
    "saturation": saturation,
    "load_shape": load_shapes,
    "fuel_share": fuel_share,
    "start_year": 2021,
    "end_year": 2041,
    "load_shape_path": netcdf_path,
}

test_end_uses = create_end_uses(**end_use_params)

test_building = {
    "building_label": "Single Family",
    "end_uses": test_end_uses,
    "start_year": 2021,
    "end_year": 2041,
    "building_stock": np.linspace(200000, 200000, 21).tolist(),
    "segment": "Single Family",
    "construction_vintage": "Existing",
}

test_build_parsed = Building(**test_building)
test_stockturnover = BuildingModel(test_build_parsed)
