import pytest
import copy
import numpy as np

from pydantic import ValidationError
from enduse.stockobjects import Equipment, RampEfficiency, EndUse, Building, LoadShape

# TODO test LoadShape

equipment = []

equipment.append(
    {
        "equipment_label": "Below Standard Heat Pump - SEER/EER 10/9.2 and HSPF 7.2 (Split System)",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.75, 0.75, 10).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(9, 9, 10).tolist(),
    }
)

equipment.append(
    {
        "equipment_label": "Federal Standard 2015 Heat Pump - SEER/EER 14/12 and HSPF 8.2 (Split System)",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.20, 0.20, 10).tolist(),
        "unit_consumption": np.linspace(7442, 7442, 10).tolist(),
        "useful_life": np.linspace(18, 18, 10).tolist(),
    }
)

equipment.append(
    {
        "equipment_label": "HVAC Upgrade - Heat Pump Upgrade to 9.5 HSPF/15.5 SEER + HZ1CZ1",
        "efficiency_level": 3,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.05, 0.05, 10).tolist(),
        "unit_consumption": np.linspace(6442, 6442, 10).tolist(),
        "useful_life": np.linspace(18, 18, 10).tolist(),
    }
)

bad_equipment = {
    "equipment_label": "Below Standard Heat Pump - SEER/EER 10/9.2 and HSPF 7.2 (Split System)",
    "efficiency_level": 1,
    "start_year": 2022,
    "end_year": 2031,
    "efficiency_share": np.linspace(0.75, 0.75, 5).tolist(),
    "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
    "useful_life": np.linspace(9, 9, 10).tolist(),
}


class TestEquipment:
    def test_valid_equipment(self):
        assert isinstance(Equipment(**equipment[0]), Equipment)

    def test_equipment_fail_expected_list_length(self):
        with pytest.raises(ValidationError):
            Equipment(**bad_equipment)


# need to parse equipment to test ramp
# ensure that TestEquipment passes first
equipment_parsed = [Equipment(**i) for i in equipment]

valid_ramp = {
    "ramp_label": "Upgrade Heat Pump",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [equipment_parsed[1], equipment_parsed[2]],
}

ramp_no_ramp_year = {
    "ramp_label": "Upgrade Heat Pump - Bad",
    "ramp_equipment": [equipment_parsed[0]],
}

ramp_fail_year_range = {
    "ramp_label": "Upgrade Heat Pump - Bad",
    "ramp_year": [2020, 2040],
    "ramp_equipment": [equipment_parsed[0], equipment_parsed[1]],
}

ramp_fail_first_year = {
    "ramp_label": "Upgrade Heat Pump - Bad",
    "ramp_year": [2025],
    "ramp_equipment": [equipment_parsed[0]],
}

ramp_fail_list_length = {
    "ramp_label": "Upgrade Heat Pump - Bad",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [equipment_parsed[0], equipment_parsed[1], equipment_parsed[2]],
}


class TestRampEfficiency:
    def test_valid_ramp(self):
        assert isinstance(RampEfficiency(**valid_ramp), RampEfficiency)

    def test_no_ramp_year(self):
        ramp_year = getattr(RampEfficiency(**ramp_no_ramp_year), "ramp_year")[0]
        exp_ramp_year = getattr(equipment_parsed[0], "start_year")
        assert ramp_year == exp_ramp_year

    def test_ramp_fail_year_range(self):
        with pytest.raises(ValidationError):
            RampEfficiency(**ramp_fail_year_range)

    def test_ramp_fail_first_year(self):
        with pytest.raises(ValidationError):
            RampEfficiency(**ramp_fail_first_year)

    def test_ramp_fail_same_list_length(self):
        with pytest.raises(ValidationError):
            RampEfficiency(**ramp_fail_list_length)


ramp_parsed = RampEfficiency(**valid_ramp)

end_use = []

end_use.append(
    {
        "end_use_label": "Heat Pump",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
        "ramp_efficiency": ramp_parsed,
    }
)

equipment_bad_list_length = copy.deepcopy(equipment)

equipment_bad_list_length.append(
    {
        "equipment_label": "Bad Heat Pump",
        "efficiency_level": 4,
        "start_year": 2022,
        "end_year": 2026,
        "efficiency_share": np.linspace(0, 0, 5).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 5).tolist(),
        "useful_life": np.linspace(9, 9, 5).tolist(),
    }
)

bad_end_use_1 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 10).tolist(),
    "fuel_share": np.linspace(1, 1, 10).tolist(),
    "equipment": [Equipment(**i) for i in equipment_bad_list_length],
    "ramp_efficiency": ramp_parsed,
}

equipment_bad_allocation = copy.deepcopy(equipment)
equipment_bad_allocation.append(
    {
        "equipment_label": "Bad Heat Pump",
        "efficiency_level": 4,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.5, 0.5, 10).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(9, 9, 10).tolist(),
    }
)

bad_end_use_2 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 10).tolist(),
    "fuel_share": np.linspace(1, 1, 10).tolist(),
    "equipment": [Equipment(**i) for i in equipment_bad_allocation],
    "ramp_efficiency": ramp_parsed,
}

equipment_bad_levels = copy.deepcopy(equipment)
equipment_bad_levels.append(
    {
        "equipment_label": "Bad Heat Pump",
        "efficiency_level": 10,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.5, 0.5, 10).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(9, 9, 10).tolist(),
    }
)

bad_end_use_3 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 10).tolist(),
    "fuel_share": np.linspace(1, 1, 10).tolist(),
    "equipment": [Equipment(**i) for i in equipment_bad_levels],
    "ramp_efficiency": ramp_parsed,
}

equipment_bad_start_end_year = copy.deepcopy(equipment)
equipment_bad_start_end_year.append(
    {
        "equipment_label": "Bad Heat Pump",
        "efficiency_level": 4,
        "start_year": 2032,
        "end_year": 2041,
        "efficiency_share": np.linspace(0.5, 0.5, 10).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(9, 9, 10).tolist(),
    }
)

bad_end_use_4 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 10).tolist(),
    "fuel_share": np.linspace(1, 1, 10).tolist(),
    "equipment": [Equipment(**i) for i in equipment_bad_start_end_year],
    "ramp_efficiency": ramp_parsed,
}

bad_end_use_5 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 20).tolist(),
    "fuel_share": np.linspace(1, 1, 20).tolist(),
    "equipment": [Equipment(**i) for i in equipment],
    "ramp_efficiency": ramp_parsed,
}

wrong_equipment = {
    "equipment_label": "Wrong Equipment",
    "efficiency_level": 2,
    "start_year": 2022,
    "end_year": 2031,
    "efficiency_share": np.linspace(0.75, 0.75, 10).tolist(),
    "unit_consumption": np.linspace(8742, 8742, 10).tolist(),
    "useful_life": np.linspace(9, 9, 10).tolist(),
}

wrong_ramp = {"label": "Wrong Ramp", "ramp_equipment": [Equipment(**wrong_equipment)]}

bad_end_use_6 = {
    "end_use_label": "Heat Pump",
    "start_year": 2022,
    "end_year": 2031,
    "saturation": np.linspace(0.25, 0.25, 10).tolist(),
    "fuel_share": np.linspace(1, 1, 10).tolist(),
    "equipment": [Equipment(**i) for i in equipment],
    "ramp_efficiency": RampEfficiency(**wrong_ramp),
}


class TestEndUse:
    def test_valid_end_use(self):
        assert isinstance(EndUse(**end_use[0]), EndUse)

    def test_end_use_fail_equipment_list_length(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_1)

    def test_end_use_fail_equipment_allocation(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_2)

    def test_end_use_fail_efficiency_levels(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_3)

    def test_end_use_fail_start_end_years(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_4)

    def test_end_use_fail_expected_list_length(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_5)

    def test_end_use_fail_ramp(self):
        with pytest.raises(ValidationError):
            EndUse(**bad_end_use_6)


end_use_parsed = [EndUse(**i) for i in end_use]

buildings = []
buildings.append(
    {
        "building_label": "Single Family",
        "start_year": 2022,
        "end_year": 2031,
        "end_uses": end_use_parsed,
        "building_stock": np.linspace(1000, 1000, 10).tolist(),
        "segment": "Residential",
        "construction_vintage": "Existing",
    }
)

equipment_fail_end_use_list_len = []

equipment_fail_end_use_list_len.append(
    {
        "equipment_label": "Below Standard Heat Pump - SEER/EER 10/9.2 and HSPF 7.2 (Split System)",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(0.5, 0.5, 20).tolist(),
        "unit_consumption": np.linspace(8742, 8742, 20).tolist(),
        "useful_life": np.linspace(9, 9, 20).tolist(),
    }
)

equipment_fail_end_use_list_len.append(
    {
        "equipment_label": "Federal Standard 2015 Heat Pump - SEER/EER 14/12 and HSPF 8.2 (Split System)",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(0.5, 0.5, 20).tolist(),
        "unit_consumption": np.linspace(7442, 7442, 20).tolist(),
        "useful_life": np.linspace(18, 18, 20).tolist(),
    }
)

equipment_fail_end_use_list_len_parsed = [
    Equipment(**i) for i in equipment_fail_end_use_list_len
]

ramp_fail_end_use_list_len = {
    "ramp_label": "Upgrade Heat Pump - Fail List Len",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [
        equipment_fail_end_use_list_len_parsed[0],
        equipment_fail_end_use_list_len_parsed[1],
    ],
}

end_use_fail_list_length = copy.deepcopy(end_use)

end_use_fail_list_length.append(
    {
        "end_use_label": "Heat Pump",
        "start_year": 2022,
        "end_year": 2041,
        "saturation": np.linspace(0.25, 0.25, 20).tolist(),
        "fuel_share": np.linspace(1, 1, 20).tolist(),
        "equipment": equipment_fail_end_use_list_len_parsed,
        "ramp_efficiency": ramp_fail_end_use_list_len,
    }
)

building_fail_list_length = {
    "building_label": "Single Family",
    "start_year": 2022,
    "end_year": 2041,
    "end_uses": [EndUse(**i) for i in end_use_fail_list_length],
    "building_stock": np.linspace(1000, 1000, 10).tolist(),
    "segment": "Residential",
    "construction_vintage": "Existing",
}


class TestBuilding:
    def test_valid_building(self):
        assert isinstance(Building(**buildings[0]), Building)

    def test_building_fail_end_use_list_length(self):
        with pytest.raises(ValidationError):
            Building(**building_fail_list_length)


building_parsed = [Building(**i) for i in buildings]

valid_resstock_path = "I:/FINANCE/FPU/LOAD/Model Development/enduse/examples/inputs/residential/resstock_loadshapes.nc"

valid_load_shape = {
    "source_file": valid_resstock_path,
    "dim_filters": {
        "shape.type": "Load Shape",
        "in.geometry_building_type_recs": "Mobile Home",
        "in.puma": "G53011606",
    },
    "value_filter": "out.electricity.heating.energy_consumption",
    "freq": "H",
    "extra_dims": {"weather_year": 30, "forecast_year": 20},
}

load_shape_fail_path = copy.deepcopy(valid_load_shape)
load_shape_fail_path["source_file"] = "./invalid_path/no_file.nc"

load_shape_fail_extra_dims_len = copy.deepcopy(valid_load_shape)
load_shape_fail_extra_dims_len["extra_dims"] = {
    "weather_year": 10,
    "forecast_year": 10,
    "bad_dim": 5,
}


load_shape_fail_extra_dims_keys = copy.deepcopy(valid_load_shape)
load_shape_fail_extra_dims_keys["extra_dims"] = {
    "bad_weather_year": 10,
    "bad_forecast_year": 10,
}


class TestLoadShape:
    def test_valid_load_shape(self):
        assert isinstance(LoadShape(**valid_load_shape), LoadShape)

    def test_load_shape_fail_path(self):
        with pytest.raises(FileNotFoundError):
            LoadShape(**load_shape_fail_path)

    def test_load_shape_fail_extra_dim_len(self):
        with pytest.raises(ValueError):
            LoadShape(**load_shape_fail_extra_dims_len)

    def test_load_shape_fail_extra_dim_keys(self):
        with pytest.raises(ValueError):
            LoadShape(**load_shape_fail_extra_dims_keys)
