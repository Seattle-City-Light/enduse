import numpy as np

from enduse.stockobjects import Equipment, RampEfficiency, EndUse, Building
from enduse.stockturnover import (
    BuildingModel,
    _create_stock_turnover,
    _create_ramp_matrix,
)

# hotkey for vscode find/replace
# ctrl + f on selected word then alt + enter to put cursor on all results

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
    [[100, 112.5, 102.5, 94.5, 88.10], [50, 62.5, 97.5, 130.5, 161.9]]
)


# valid equipment mat with exogenous subtractions
valid_equip_mat_subs = np.array([np.linspace(100, 75, 5), np.linspace(50, 25, 5)])
exp_valid_st_mat_subs = np.array(
    [[100, 93.75, 68.75, 48.75, 32.75], [50, 43.75, 56.25, 63.75, 67.25]]
)


class TestStockTurnoverCalculation:
    def test_valid_stock_turnover_with_ramp(self):
        calc_stock_turn = _create_stock_turnover(
            valid_equip_mat, valid_ul_mat, valid_ramp_mat
        )
        # check to make sure that calculated match expected
        assert np.array_equal(calc_stock_turn, exp_valid_st_mat)
        # check to make sure that equip_mat totals have been preserved
        assert np.array_equal(
            np.sum(calc_stock_turn, axis=0), np.sum(valid_equip_mat, axis=0)
        )

    def test_valid_ramp_mat_ones(self):
        calc_stock_turn = _create_stock_turnover(
            valid_equip_mat, valid_ul_mat, valid_ramp_mat_ones
        )
        assert np.array_equal(calc_stock_turn, valid_equip_mat)

    def test_valid_st_mat_adds(self):
        calc_stock_turn = _create_stock_turnover(
            valid_equip_mat_adds, valid_ul_mat, valid_ramp_mat
        )
        assert np.array_equal(
            np.round(calc_stock_turn, 0), np.round(exp_valid_st_mat_adds, 0)
        )

    def test_valid_st_mat_subs(self):
        calc_stock_turn = _create_stock_turnover(
            valid_equip_mat_subs, valid_ul_mat, valid_ramp_mat
        )
        assert np.array_equal(
            np.round(calc_stock_turn, 0), np.round(exp_valid_st_mat_subs, 0)
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
        "unit_consumption": np.linspace(8000, 8000, 10).tolist(),
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
        "unit_consumption": np.linspace(7000, 7000, 10).tolist(),
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
        "unit_consumption": np.linspace(5000, 5000, 10).tolist(),
        "useful_life": np.linspace(20, 20, 10).tolist(),
    }
)


valid_equipment_parsed = [Equipment(**i) for i in valid_equipment]

valid_multi_ramp = {
    "ramp_label": "Upgrade Heat Pump",
    "ramp_year": [2022, 2025],
    "ramp_equipment": [valid_equipment_parsed[1], valid_equipment_parsed[2]],
}

valid_multi_ramp_parsed = RampEfficiency(**valid_multi_ramp)

valid_single_ramp = {
    "ramp_label": "No Ramp",
    "ramp_equipment": [valid_equipment_parsed[2]],
}

valid_single_ramp_parsed = RampEfficiency(**valid_single_ramp)

expected_multi_ramp_matrix = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

expected_single_ramp_matrix = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)


class TestCreateRampMatix:

    equip_mat = np.tile(np.linspace(100, 100, 10), (3, 1))

    def test_ramp_matrix_multiple_ramps(self):
        ramp_matrix = _create_ramp_matrix(self.equip_mat, valid_multi_ramp_parsed)
        assert np.array_equal(ramp_matrix, expected_multi_ramp_matrix)

    def test_ramp_matrix_single_ramp(self):
        ramp_matrix = _create_ramp_matrix(self.equip_mat, valid_single_ramp_parsed)
        assert np.array_equal(ramp_matrix, expected_single_ramp_matrix)


valid_end_uses = []
valid_end_uses.append(
    {
        "end_use_label": "Heat Pump 1",
        "start_year": 2022,
        "end_year": 2031,
        "equipment": valid_equipment_parsed,
        "ramp_efficiency": valid_multi_ramp_parsed,
        "saturation": np.linspace(0.50, 0.50, 10).tolist(),
        "fuel_share": np.linspace(0.50, 0.50, 10).tolist(),
    }
)

valid_equipment_2 = []
valid_equipment_2.append(
    {
        "equipment_label": "Below Standard Heat Pump",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.50, 0.50, 10).tolist(),
        "unit_consumption": np.linspace(8000, 8000, 10).tolist(),
        "useful_life": np.linspace(5, 15, 10).tolist(),
    }
)

valid_equipment_2.append(
    {
        "equipment_label": "Standard Heat Pump",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.50, 0.50, 10).tolist(),
        "unit_consumption": np.linspace(7000, 7000, 10).tolist(),
        "useful_life": np.linspace(18, 18, 10).tolist(),
    }
)

valid_end_uses.append(
    {
        "end_use_label": "Heat Pump 2",
        "start_year": 2022,
        "end_year": 2031,
        "equipment": valid_equipment_2,
        "saturation": np.linspace(0.50, 0.50, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
    }
)

valid_end_use_parsed = [EndUse(**i) for i in valid_end_uses]

valid_building = {
    "building_label": "Test Building",
    "start_year": 2022,
    "end_year": 2031,
    "end_uses": valid_end_use_parsed,
    "building_stock": np.linspace(1000, 1000, 10).tolist(),
}

valid_building_parsed = Building(**valid_building)

stock_turnover = BuildingModel(valid_building_parsed)


class TestBuildingModel:

    sel = {"init_end_use_label": "Heat Pump 1"}

    def test_xarray_dims(self):

        expected_dims = {"init_end_use_label": 2, "year": 10, "efficiency_level": 3}

        assert expected_dims == stock_turnover.model.dims

    def test_xarray_coords_keys(self):

        expected_coords_keys = [
            "efficiency_level",
            "efficiency_label",
            "year",
            "end_use_label",
            "init_end_use_label",
            "building_label",
        ]

        assert expected_coords_keys == list(stock_turnover.model.coords.keys())

    def test_xarray_data_vars(self):
        expected_data_vars = [
            "building_stock",
            "saturation",
            "fuel_share",
            "ramp_matrix",
            "efficiency_share",
            "unit_consumption",
            "useful_life",
            "init_equipment_stock",
            "equipment_stock",
            "consumption",
        ]
        assert list(stock_turnover.model.data_vars.keys()) == expected_data_vars

    def test_xarray_equipment_stock(self):
        expected_stock = np.linspace(250, 250, 10)
        calc_stock = np.sum(
            stock_turnover.model.sel(self.sel)["equipment_stock"], axis=0
        )
        assert np.array_equal(np.round(expected_stock, 0), np.round(calc_stock, 0))

    def test_xarray_consumptions(self):
        calc_stock = stock_turnover.model.sel(self.sel)["equipment_stock"]
        exp_cons = np.array([8000, 7000, 5000])
        exp_cons = calc_stock * exp_cons[:, None]
        calc_cons = stock_turnover.model.sel(self.sel)["consumption"]
        assert np.array_equal(np.round(exp_cons, 0), np.round(calc_cons, 0))
