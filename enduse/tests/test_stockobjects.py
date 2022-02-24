import pytest
import copy

from pydantic import ValidationError
from enduse import stockobjects

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
    "join_order": 0,
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
        "Year",
    ],
    "join_order": 4,
}

params["ramp_efficiency"] = {
    "label": "Efficiency Ramp",
    "file_path": path + "equipment_standards.csv",
    "time_label": "Year",
    "interpolation_label": "Interpolate Efficiency Ramp Probability",
    "start_variable": "Start Efficiency Ramp Probability",
    "end_variable": "End Efficiency Ramp Probability",
    "start_time": "Start Year",
    "end_time": "End Year",
    "join_on": [
        "Customer Class",
        "Segment",
        "End Use",
        "Ramp Efficiency Level",
        "Year",
    ],
    "join_order": 5,
}

bad_params = copy.deepcopy(params)

bad_params["saturation"]["start_time"] = 12


class TestStockObject:
    def test_attributes_loaded(self):
        """Check that all children classes are loaded from dict"""
        stock_object = stockobjects.StockObject.parse_obj(params)
        assert stock_object.dict() == params

    def test_bad_attribute(self):
        """Check that a bad attribute raises a pydantic error"""
        with pytest.raises(ValidationError):
            stockobjects.StockObject.parse_obj(bad_params)
