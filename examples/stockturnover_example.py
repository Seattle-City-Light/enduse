import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from enduse.stockobjects import (
    Equipment,
    RampEfficiency,
    EndUse,
    Building,
    LoadShape,
)

from enduse.stockturnover import BuildingModel

# set load shape path
resstock_path = str(
    (Path(__file__).parents[1] / "outputs/loadshapes/resstock_loadshapes.nc").as_posix()
)

# heat central
heat_central_equip = []

heat_central_equip.append(
    {
        "equipment_label": "Standard Electric Furnace HSPF = 1",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(1, 1, 20).tolist(),
        "unit_consumption": np.linspace(13312, 13312, 20).tolist(),
        "useful_life": np.linspace(15, 15, 20).tolist(),
    }
)

heat_central_equip.append(
    {
        "equipment_label": "High Efficiency Electric Furnace",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(0, 0, 20).tolist(),
        "unit_consumption": np.linspace(8500, 8500, 20).tolist(),
        "useful_life": np.linspace(18, 18, 20).tolist(),
    }
)

heat_central_equip_parsed = [Equipment(**i) for i in heat_central_equip]

heat_central_ramp = {
    "ramp_label": "Forced Air Furnance High Efficiency Upgrade",
    "ramp_equipment": [heat_central_equip[1]],
}

heat_central_ramp_parsed = RampEfficiency(**heat_central_ramp)

heat_central_shape = {
    "source_file": resstock_path,
    "dim_filters": {
        "shape.type": "Load Shape",
        "in.puma": "G53011606",
        "in.geometry_building_type_recs": "Single-Family Detached",
    },
    "value_filter": "out.electricity.heating.energy_consumption",
    "freq": "H",
}

end_uses = []

end_uses.append(
    {
        "end_use_label": "Heat Central",
        "equipment": heat_central_equip_parsed,
        "ramp_efficiency": heat_central_ramp_parsed,
        "saturation": np.linspace(0.50, 0.50, 20).tolist(),
        "fuel_share": np.linspace(0.05, 0.05, 20).tolist(),
        "load_shape": LoadShape(**heat_central_shape),
    }
)

# cooking range
cook_range_equip = {
    "equipment_label": "Federal Standard Cooking Range",
    "efficiency_level": 1,
    "start_year": 2022,
    "end_year": 2041,
    "efficiency_share": np.linspace(1, 1, 20).tolist(),
    "unit_consumption": np.linspace(125, 125, 20).tolist(),
    "useful_life": np.linspace(19, 19, 20).tolist(),
}

cook_range_ls = {
    "source_file": resstock_path,
    "dim_filters": {
        "shape.type": "Load Shape",
        "in.puma": "G53011606",
        "in.geometry_building_type_recs": "Single-Family Detached",
    },
    "value_filter": "out.electricity.cooking_range.energy_consumption",
    "freq": "H",
}

end_uses.append(
    {
        "end_use_label": "Cooking Range",
        "equipment": [Equipment(**cook_range_equip)],
        "saturation": np.linspace(1, 1, 20).tolist(),
        "fuel_share": np.linspace(0.25, 0.25, 20).tolist(),
        "load_shape": LoadShape(**cook_range_ls),
    }
)

# end use no shape
heat_pump_ls = {
    "source_file": resstock_path,
    "dim_filters": {
        "shape.type": "Load Shape",
        "in.puma": "G53011606",
        "in.geometry_building_type_recs": "Single-Family Detached",
    },
    "value_filter": "agg.electric.heat_pump.energy_consumption",
    "freq": "H",
}

other_equip = {
    "equipment_label": "Other Equipment",
    "efficiency_level": 1,
    "start_year": 2022,
    "end_year": 2041,
    "efficiency_share": np.linspace(1, 1, 20).tolist(),
    "unit_consumption": np.linspace(250, 250, 20).tolist(),
    "useful_life": np.linspace(5, 5, 20).tolist(),
}

end_uses.append(
    {
        "end_use_label": "Other Equipment",
        "equipment": [Equipment(**other_equip)],
        "saturation": np.linspace(1, 1, 20).tolist(),
        "fuel_share": np.linspace(1, 1, 20).tolist(),
    }
)

# end use with multiple load shapes
heat_pump_ls = {
    "source_file": resstock_path,
    "dim_filters": {
        "shape.type": "Load Shape",
        "in.puma": "G53011606",
        "in.geometry_building_type_recs": "Single-Family Detached",
    },
    "value_filter": "agg.electric.heat_pump.energy_consumption",
    "freq": "H",
}


heat_central_multi_shape_equip = []

heat_central_multi_shape_equip.append(
    {
        "equipment_label": "Standard Electric Furnace HSPF = 1",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(1, 1, 20).tolist(),
        "unit_consumption": np.linspace(13312, 13312, 20).tolist(),
        "useful_life": np.linspace(15, 15, 20).tolist(),
    }
)

heat_central_multi_shape_equip.append(
    {
        "equipment_label": "Install Ductless Heat Pump in House with Existing FAF - HZ1",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(0, 0, 20).tolist(),
        "unit_consumption": np.linspace(8500, 8500, 20).tolist(),
        "useful_life": np.linspace(18, 18, 20).tolist(),
        "load_shape": LoadShape(**heat_pump_ls),
    }
)

heat_central_multi_shape_equip_parsed = [
    Equipment(**i) for i in heat_central_multi_shape_equip
]

heat_central_multi_ramp = {
    "ramp_label": "Heat Pump Efficiency Upgrade",
    "ramp_equipment": [
        heat_central_multi_shape_equip[0],
        heat_central_multi_shape_equip[1],
    ],
    "ramp_year": [2022, 2025],
}

heat_central_multi_ramp_parsed = RampEfficiency(**heat_central_multi_ramp)

end_uses.append(
    {
        "end_use_label": "Heat Central Multi Shape",
        "equipment": heat_central_multi_shape_equip_parsed,
        "ramp_efficiency": heat_central_multi_ramp_parsed,
        "saturation": np.linspace(0.25, 0.25, 20).tolist(),
        "fuel_share": np.linspace(0.25, 0.25, 20).tolist(),
        "load_shape": LoadShape(**heat_central_shape),
    }
)

building = {
    "building_label": "Single Family",
    "end_uses": [EndUse(**i) for i in end_uses],
    "building_stock": np.linspace(200000, 200000, 20).tolist(),
}

building_parsed = Building(**building)
stock_turnover = BuildingModel(building_parsed)

# TODO create logic where single equipment can override end-use load shape
# TODO need to add attrs to end_use to validate that equipment load shape is consistent with end_use
