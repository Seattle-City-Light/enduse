import numpy as np
import pytest

from enduse.stockobjects import Equipment, RampEfficiency, EndUse, Building
from enduse.stockturnover import (
    get_building_arrays,
    stock_turnover_calculation,
    create_ramp_matrix,
)


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


# need to create Equipment and RampEfficiency objects to test ramp
valid_equipment = []

valid_equipment.append(
    {
        "equipment_label": "Below Standard Heat Pump",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.75, 0.75, 10).tolist(),
        "consumption": np.linspace(8742, 8742, 10).tolist(),
        "useful_life": np.linspace(5, 15, 10).tolist(),
    }
)

valid_equipment.append(
    {
        "equipment_label": "Standard Heat Pump",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.20, 0.20, 10).tolist(),
        "consumption": np.linspace(7442, 7442, 10).tolist(),
        "useful_life": np.linspace(18, 18, 10).tolist(),
    }
)

valid_equipment.append(
    {
        "equipment_label": "Above Standard Heat Pump",
        "efficiency_level": 3,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.05, 0.05, 10).tolist(),
        "consumption": np.linspace(5000, 5000, 10).tolist(),
        "useful_life": np.linspace(20, 20, 10).tolist(),
    }
)


valid_equipment_parsed = [Equipment(**i) for i in valid_equipment]

valid_ramp = {
    "ramp_label": "Upgrade Heat Pump",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [valid_equipment_parsed[1], valid_equipment_parsed[2]],
}

valid_ramp_parsed = RampEfficiency(**valid_ramp)

expected_ramp_matrix = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)


def test_valid_create_ramp_matrix(self):
    equip_mat = np.tile(np.linspace(100, 100, 10), (3, 1))
    ramp_matrix = create_ramp_matrix(self.equip_mat, valid_ramp_parsed)
    assert np.array_equal(ramp_matrix, expected_ramp_matrix)
