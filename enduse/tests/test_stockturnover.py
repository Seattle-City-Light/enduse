import numpy as np
import pytest

from enduse.stockobjects import Equipment, RampEfficiency, EndUse, Building
from enduse.stockturnover import get_building_arrays, stock_turnover_calculation


# define valid test case for stock turnover
# note st_mats are calculated by hand
valid_equip_mat = np.array([np.linspace(100, 100, 5), np.linspace(50, 50, 5)])
valid_ul_mat = np.array([np.linspace(5, 5, 5), np.linspace(10, 10, 5)])
valid_ramp_mat = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
exp_valid_st_mat = np.array([[100, 100, 80, 64, 51.2], [50, 50, 70, 86, 98.8]])

# valid case with no ramp specified - no change in equipment
valid_ramp_mat_ones = np.ones(valid_ramp_mat.shape)

# valid equipment mat with exogenous additions
valid_equip_mat_adds = np.array([np.linspace(100, 150, 5), np.linspace(50, 100, 5)])
exp_valid_st_mat_adds = np.array(
    [[100, 112.5, 102.5, 94.5, 88.1], [50, 62.5, 97.5, 130.5, 161.9]]
)

# valid equipment mat with exogenous subtractions
valid_equip_mat_subs = np.array([np.linspace(100, 25, 5), np.linspace(50, 25, 5)])
exp_valid_st_mat_subs = np.array(
    [[100, 81.25, 46.25, 18.25, -4.15], [50, 43.75, 53.75, 56.75, 54.15]]
)


class TestStockTurnoverCalculation:
    def test_valid_stock_turnover_with_ramp(self):
        calc_stock_turn = stock_turnover_calculation(
            valid_equip_mat, valid_ul_mat, valid_ramp_mat
        )
        # check to make sure that calculated match expected
        assert np.array_equal(calc_stock_turn, exp_valid_st_mat)
        # check to make sure that equip_mat totals have been preserved
        assert np.array_equal(
            np.sum(calc_stock_turn, axis=0), np.sum(valid_equip_mat, axis=0)
        )

    def test_valid_ramp_mat_ones(self):
        calc_stock_turn = stock_turnover_calculation(
            valid_equip_mat, valid_ul_mat, valid_ramp_mat_ones
        )
        assert np.array_equal(calc_stock_turn, valid_equip_mat)
        assert np.array_equal(
            np.sum(calc_stock_turn, axis=0), np.sum(valid_equip_mat, axis=0)
        )

    def test_valid_st_mat_adds(self):
        calc_stock_turn = stock_turnover_calculation(
            valid_equip_mat_adds, valid_ul_mat, valid_ramp_mat
        )
        assert np.array_equal(calc_stock_turn, exp_valid_st_mat_adds)
        assert np.array_equal(
            np.sum(calc_stock_turn, axis=0), np.sum(valid_equip_mat_adds, axis=0)
        )

    def test_valid_st_mat_subs(self):
        calc_stock_turn = stock_turnover_calculation(
            valid_equip_mat_subs, valid_ul_mat, valid_ramp_mat
        )
        assert np.array_equal(calc_stock_turn, exp_valid_st_mat_subs)
        assert np.array_equal(
            np.sum(calc_stock_turn, axis=0), np.sum(valid_equip_mat_subs, axis=0)
        )


# for testing xarray creation

equipment = []

equipment.append(
    {
        "equipment_label": "Below Standard Heat Pump - SEER/EER 10/9.2 and HSPF 7.2 (Split System)",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.75, 0.75, 10).tolist(),
        "consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(5, 15, 10).tolist(),
    }
)

equipment.append(
    {
        "equipment_label": "Federal Standard 2015 Heat Pump - SEER/EER 14/12 and HSPF 8.2 (Split System)",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.20, 0.20, 10).tolist(),
        "consumption": np.linspace(7442, 7442, 10).tolist(),
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
        "consumption": np.linspace(6442, 6442, 10).tolist(),
        "useful_life": np.linspace(18, 18, 10).tolist(),
    }
)

single_equipment = {
    "equipment_label": "HVAC Upgrade - Heat Pump Upgrade to 9.5 HSPF/15.5 SEER + HZ1CZ1",
    "efficiency_level": 1,
    "start_year": 2022,
    "end_year": 2031,
    "efficiency_share": np.linspace(1, 1, 10).tolist(),
    "consumption": np.linspace(6442, 6442, 10).tolist(),
    "useful_life": np.linspace(18, 18, 10).tolist(),
}

single_equipment_parsed = Equipment(**single_equipment)

equipment_parsed = [Equipment(**i) for i in equipment]

ramp = {
    "ramp_label": "Upgrade Heat Pump",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [equipment_parsed[1], equipment_parsed[2]],
}

ramp_single_equipment = {
    "ramp_label": "Upgrade Heat Pump",
    "ramp_equipment": [single_equipment_parsed],
}

ramp_parsed = RampEfficiency(**ramp)
ramp_single_parsed = RampEfficiency(**ramp_single_equipment)

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

end_use.append(
    {
        "end_use_label": "Heat Pump - Exogenous Additions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.50, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
        "ramp_efficiency": ramp_parsed,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - Exogenous Subtractions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.50, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
        "ramp_efficiency": ramp_parsed,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - Single Equipment",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": [single_equipment_parsed],
        "ramp_efficiency": ramp_single_equipment,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - Single Equipment Exogenous Additions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.50, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": [single_equipment_parsed],
        "ramp_efficiency": ramp_single_equipment,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - Single Equipment Exogenous Subtractions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.50, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": [single_equipment_parsed],
        "ramp_efficiency": ramp_single_equipment,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - No Ramp",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.50, 0.50, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - No Ramp Exogenous Additions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.50, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
    }
)

end_use.append(
    {
        "end_use_label": "Heat Pump - No Ramp Exogenous Subtractions",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.50, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
    }
)

end_use_parsed = [EndUse(**i) for i in end_use]

building = {
    "building_label": "Single Family",
    "end_uses": end_use_parsed,
    "building_stock": np.linspace(1000, 1000, 10).tolist(),
    "segment": "Residential",
    "construction_vintage": "Existing",
}

building_parsed = Building(**building)

test = get_building_arrays(building_parsed)
