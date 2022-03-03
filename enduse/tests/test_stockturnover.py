import os
import pytest
import pandas as pd
import numpy as np

from enduse import stockobjects, stockturnover

### Build valid stockobject params
path = "I:/FINANCE/FPU/LOAD/2021/Model Dev/enduse/examples/inputs/residential/"

params = {}

params["buildings"] = {
    "label": "Buildings",
    "file_path": path + "building_stock.csv",
    "join_on": ["Customer Class", "Segment", "Construction Vintage", "Year"],
    "join_order": 0,
}

params["saturation"] = {
    "label": "Saturation",
    "file_path": path + "saturation.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Saturation",
    "start_variable": "Start Saturation",
    "end_variable": "End Saturation",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": ["Customer Class", "Segment", "Construction Vintage", "End Use", "Year"],
    "join_order": 1,
}

params["fuel_share"] = {
    "label": "Fuel Share",
    "file_path": path + "fuel_share.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Fuel Share",
    "start_variable": "Start Fuel Share",
    "end_variable": "End Fuel Share",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": ["Customer Class", "Segment", "Construction Vintage", "End Use", "Year"],
    "join_order": 2,
}

params["efficiency_share"] = {
    "label": "Efficiency Share",
    "file_path": path + "efficiency_share.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Efficiency Share",
    "start_variable": "Start Efficiency Share",
    "end_variable": "End Efficiency Share",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": ["Customer Class", "Segment", "Construction Vintage", "End Use", "Year"],
    "join_order": 3,
}

params["equipment_measures"] = {
    "label": "Measure Consumption",
    "file_path": path + "equipment_measures_adjusted.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Measure Consumption",
    "start_variable": "Start Measure Consumption",
    "end_variable": "End Measure Consumption",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": [
        "Customer Class",
        "Segment",
        "Construction Vintage",
        "End Use",
        "Efficiency Level",
        "Efficiency Description",
        "Year",
    ],
    "join_order": 4,
}

params["ramp_efficiency"] = {
    "label": "Ramp Efficiency Probability",
    "file_path": path + "equipment_standards_mapped.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Ramp Efficiency Probability",
    "start_variable": "Start Ramp Efficiency Probability",
    "end_variable": "End Ramp Efficiency Probability",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": ["Customer Class", "Segment", "End Use", "Year"],
    "join_order": 5,
}

calc_config_params = {
    "equipment_label": "Equipment",
    "equipment_stock_label": "Equipment Stock",
    "equipment_consumption_label": "Consumption",
    "equipment_measure_consumption_label": "Measure Consumption",
    "equipment_sort_index": [
        "Customer Class",
        "Segment",
        "Construction Vintage",
        "End Use",
        "Efficiency Level",
    ],
    "equipment_calc_variables": [
        "Building Stock",
        "Saturation",
        "Fuel Share",
        "Efficiency Share",
    ],
    "equipment_calc_index": [
        "Customer Class",
        "Segment",
        "Construction Vintage",
        "End Use",
    ],
    "effective_useful_life_label": "Effective Useful Life",
    "efficiency_level_label": "Efficiency Level",
    "ramp_efficiency_level_label": "Ramp Efficiency Level",
    "ramp_efficiency_probability_label": "Ramp Efficiency Probability",
}

# create dummy dataframes
saturation_test = pd.DataFrame(
    {
        "Segment": ["Multifamily - High Rise", "Single Family"],
        "End Use": ["Heat Room", "Heat Central"],
        "Construction Vintaege": ["Existing", "New"],
        "Start Saturation": [0.5, 1],
        "End Saturation": [1, 1],
        "Start Year": [2021, 2021],
        "End Year": [2041, 2041],
        "Interpolate Saturation": ["linear", "linear"],
        "Customer Class": ["Residential", "Residential"],
    }
)

ramp_efficiency_test = pd.DataFrame(
    {
        "Segment": ["Single Family", "Single Family"],
        "End Use": ["Heat Pump", "Heat Pump"],
        "Ramp Efficiency Level": [2, 3],
        "Ramp Efficiency Description": [
            "Federal Standard 2015 Heat Pump - SEER/EER 14/12 and HSPF 8.2 (Split System)",
            "Federal Standard 2023 Heat Pump - SEER/EER 15/12.5 and HSPF 8.8 (Split System)",
        ],
        "Start Year": [2021, 2023],
        "End Year": [2022, 2041],
        "Start Ramp Efficiency Probability": [1, 1],
        "End Ramp Efficiency Probability": [1, 1],
        "Interpolate Ramp Efficiency Probability": ["linear", "linear"],
        "Customer Class": ["Residential", "Residential"],
    }
)

# parse all stock objects
stock_object = stockobjects.StockObject.parse_obj(params)

# parse calc config
calc_config = stockobjects.CalcConfig.parse_obj(calc_config_params)

# hold common files test data frames
# use interpolation value for dict keys
common_files_test = {}

common_files_test["saturation"] = saturation_test
common_files_test["ramp_efficiency"] = ramp_efficiency_test

# hold common files parsed dataframes
common_files_parsed = {}

common_files_parsed["saturation"] = stockturnover.create_common_files_dataframe(
    stock_object.saturation, saturation_test.copy()
)

common_files_parsed["ramp_efficiency"] = stockturnover.create_common_files_dataframe(
    stock_object.ramp_efficiency, ramp_efficiency_test.copy()
)


class TestCreateCommonFilesDataFrame:
    def test_create_common_files_shape(self):
        """Check that test dataframes have correct shape"""

        expected_shape = []

        # iterate over each data frame
        for n, df in common_files_test.items():

            expected_rows = 0

            # iterate over each row in dataframe
            for i, x in df.iterrows():

                # expected shape of interpolated dataframe
                year_count = x[params[n]["end_time"]] - x[params[n]["start_time"]] + 1

                expected_rows += year_count

            expected_shape.append((expected_rows, df.shape[1] - 2))

        parsed_shape = [i.shape for i in common_files_parsed.values()]

        assert expected_shape == parsed_shape

    def test_create_common_files_interpolation(self):
        """Check that interpolation is being calculated correctly"""

        expected_interp = []

        for n, df in common_files_test.items():
            for i, x in df.iterrows():

                year_count = x[params[n]["end_time"]] - x[params[n]["start_time"]] + 1

                expected_interp.append(
                    np.linspace(
                        x[params[n]["start_variable"]],
                        x[params[n]["end_variable"]],
                        year_count,
                    )
                )

        expected_interp = np.concatenate(expected_interp)

        parsed_interp = np.concatenate(
            [
                common_files_parsed[i][params[i]["label"]].values
                for i in common_files_parsed.keys()
            ]
        )

        assert expected_interp.tolist() == parsed_interp.tolist()


stock_turnover = stockturnover.StockTurnoverModel(stock_object, calc_config)
