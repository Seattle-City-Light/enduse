# import dataclasses
import json

from typing import List, Dict
from pydantic import BaseModel, StrictStr, StrictInt

# pydantic will not force validation of str or int types on class instantiation
# default pyndantic behavior is to coerce str or int to type hint
# can use StrictStr or StrictInt to force validation
class BuildingObject(BaseModel):
    """
    Container for building object which has unique input format
    """
    label: StrictStr
    file_path: StrictStr
    join_on: List[StrictStr]
    join_order: StrictInt

class CommonStockObject(BaseModel):
    """
    Container for common stock objects which share input format
    """
    label: StrictStr
    file_path: StrictStr
    time_label: StrictStr
    interpolation_label: StrictStr
    start_variable: StrictStr
    end_variable: StrictStr
    start_time: StrictStr
    end_time: StrictStr
    join_on: List[StrictStr]
    join_order: StrictInt

class StockObject(BaseModel):
    """
    Container class for multiple stock objects
    """
    buildings : BuildingObject
    saturation: CommonStockObject
    fuel_share: CommonStockObject
    efficiency_share: CommonStockObject
    equipment_measures: CommonStockObject
    ramp_efficiency : CommonStockObject

    def dump_to_json(self, path:str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, ensure_ascii=False, indent=4)

class CalcConfig(BaseModel):
    """
    Container class to hold model config variables
    """
    equipment_label: StrictStr
    equipment_stock_label: StrictStr
    equipment_consumption_label: StrictStr
    equipment_measure_consumption_label: StrictStr
    effective_useful_life_label: StrictStr
    efficiency_level_label: StrictStr
    ramp_efficiency_level_label: StrictStr
    ramp_efficiency_probability_label: StrictStr

    equipment_sort_index: List[StrictStr]
    equipment_calc_variables: List[StrictStr]
    equipment_calc_index: List[StrictStr]