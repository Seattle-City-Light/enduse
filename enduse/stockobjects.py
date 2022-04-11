import numpy as np

from typing import List, Optional, Dict
from pathlib import Path

from pydantic import (
    BaseModel,
    StrictStr,
    PositiveInt,
    PositiveFloat,
    confloat,
    Field,
    validator,
    root_validator,
    PrivateAttr,
)

# TODO alias on all field names for DF conversion?
# TODO modify ramp to handle multiple efficiency ramp levels
# TODO transition matrix could be used to handle multiple efficiency levels
# TODO modify stock objects to handle different adoption logic: type='decay', type='logit', type='markov chain'


def check_expected_list_length(v: list, values: dict):
    label = values.get("label")
    exp_len = values.get("end_year") - values.get("start_year") + 1
    if len(v) != exp_len:
        raise ValueError(f"{label} has list len {len(v)} but expected len {exp_len}")
    return v


# TODO change label consumption -> unit_consumption
class Equipment(BaseModel):
    label: StrictStr = Field(None, alias="equipment_label")
    efficiency_level: PositiveInt
    start_year: PositiveInt
    end_year: PositiveInt
    efficiency_share: List[confloat(ge=0, le=1)]
    consumption: List[PositiveFloat]
    useful_life: List[PositiveInt]

    # check list lengths match expected
    _check_expected_list_length: classmethod = validator(
        "efficiency_share", "consumption", "useful_life", allow_reuse=True
    )(check_expected_list_length)


class RampEfficiency(BaseModel):
    label: StrictStr = Field(None, alias="ramp_label")
    ramp_equipment: List[Equipment]
    ramp_year: Optional[List[PositiveInt]] = None

    # inherit ramp_year if not provided
    @validator("ramp_year", always=True)
    def set_ramp_year(cls, v, values):
        if v is None:
            return [(getattr(values.get("ramp_equipment")[0], "start_year"))]
        else:
            return v

    # check that ramp year fall within equipment year range
    @validator("ramp_year")
    def validate_ramp_year_in_range(cls, v, values):
        label = values.get("label")
        for i, x in zip(v, values.get("ramp_equipment")):
            if i < getattr(x, "start_year") or i > getattr(x, "end_year"):
                raise ValueError(f"{label} ramp_year outside of equipment year range")
        return v

    # if ramp year is provided check that first ramp equals first year in equipment
    @validator("ramp_year")
    def validate_first_ramp_year(cls, v, values):
        label = values.get("label")
        exp_min_ramp_year = getattr(values.get("ramp_equipment")[0], "start_year")
        if min(v) > exp_min_ramp_year:
            raise ValueError(
                f"{label} has min ramp_year of {v} but expected {exp_min_ramp_year}"
            )
        return v

    # validate that multiple ramp years are sequential
    @validator("ramp_year")
    def validate_sequential_ramp_year(cls, v, values):
        label = values.get("label")
        if len(v) > 1:
            for i, x in enumerate(v):
                if i > 0:
                    if v[i] <= v[i - 1]:
                        raise ValueError(f"{label} invalid ramp_year order")
        return v

    # check that ramp inputs are a consistent length
    @root_validator
    def validate_same_list_length(cls, values):
        label = values.get("label")
        length = len(values.get("ramp_year"))
        list_fields = ["ramp_year", "ramp_equipment"]
        if any(len(values[i]) != length for i in list_fields):
            raise ValueError(f"{label} list fields do not have the same length")
        return values


class LoadShape(BaseModel):
    label: StrictStr = Field(None, alias="load_shape_label")
    source_file: StrictStr
    dim_filters: Dict[StrictStr, StrictStr]
    value_filter: StrictStr

    @validator("source_file")
    def validate_source_file(cls, v):
        if not Path(v).is_file():
            raise FileNotFoundError(f"{v} is invalid path")
        return v


class EndUse(BaseModel):
    label: StrictStr = Field(None, alias="end_use_label")
    equipment: List[Equipment]
    ramp_efficiency: Optional[RampEfficiency] = None
    start_year: Optional[PositiveInt] = None
    end_year: Optional[PositiveInt] = None
    saturation: List[PositiveFloat]
    fuel_share: List[confloat(ge=0, le=1)]
    load_shape: Optional[LoadShape]

    # check that all equipment has same list length
    @validator("equipment")
    def validate_equipment_list_length(cls, v, values):
        label = values.get("label")
        length = len(getattr(v[0], "efficiency_share"))
        for i in v:
            if len(getattr(i, "efficiency_share")) != length:
                raise ValueError(
                    f"{label} equipment efficiency_share do not have the same length"
                )
        return v

    # check that equipment allocation each yeah sums to 100%
    @validator("equipment")
    def validate_equipment_allocation(cls, v, values):
        label = values.get("label")
        allocations = np.array([getattr(i, "efficiency_share") for i in v])
        if np.any(allocations.sum(axis=0) != 1):
            raise ValueError(
                f"{label} equipment efficiency_share allocations do not sum to 1"
            )
        return v

    # check that equipment has sequential efficiency levels
    @validator("equipment")
    def validate_equipment_efficeincy_levels(cls, v, values):
        label = values.get("label")
        levels = [getattr(i, "efficiency_level") for i in v]
        expected_levels = [*range(min(levels), max(levels) + 1)]
        if levels != expected_levels:
            raise ValueError(f"{label} equipment efficiency_levels are not sequential")
        return v

    # check equipment has consistent start and end years
    @validator("equipment")
    def validate_equipment_start_end_years(cls, v, values):
        label = values.get("label")
        start_year = [getattr(i, "start_year") for i in v]
        end_year = [getattr(i, "end_year") for i in v]
        if all(x != start_year[0] for x in start_year):
            raise ValueError(f"{label} equipment start_year values not equal")
        if all(y != end_year[0] for y in end_year):
            raise ValueError(f"{label} equipment end_year values not equal")
        return v

    # set start year if not provided
    @validator("start_year", always=True)
    def set_start_year(cls, v, values):
        if v is None:
            return getattr(values.get("equipment")[0], "start_year")
        else:
            return v

    # set end year if not provided
    @validator("end_year", always=True)
    def set_end_year(cls, v, values):
        if v is None:
            return getattr(values.get("equipment")[0], "end_year")
        else:
            return v

    @validator("ramp_efficiency")
    def check_ramp_equipment_valid(cls, v, values):
        label = values.get("label")
        ramp_labels = [getattr(i, "label") for i in getattr(v, "ramp_equipment")]
        equipment_labels = [getattr(i, "label") for i in values.get("equipment")]
        if not any([i in equipment_labels for i in ramp_labels]):
            raise ValueError(f"{label} ramp_equipment not in equipment")
        return v

    _check_expected_list_length: classmethod = validator(
        "saturation", "fuel_share", allow_reuse=True
    )(check_expected_list_length)


class Building(BaseModel):
    label: StrictStr = Field(None, alias="building_label")
    end_uses: List[EndUse]
    start_year: Optional[PositiveInt]
    end_year: Optional[PositiveInt]
    building_stock: List[PositiveFloat]
    segment: Optional[StrictStr] = None
    construction_vintage: Optional[StrictStr] = None

    # private attribute to track max # of efficiency shares
    # need this dimension to ensure all numpy arrays in enduse -> stockturnover have same dims
    # xarray requires DataSets to have same dims for most opertations
    _end_use_len: int = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._end_use_len = max([len(getattr(i, "equipment")) for i in self.end_uses])

    @validator("end_uses")
    def validate_end_use_list_length(cls, v, values):
        label = values.get("label")
        saturation_length = len(getattr(v[0], "saturation"))
        fuel_share_length = len(getattr(v[0], "fuel_share"))
        for i in v:
            if len(getattr(i, "saturation")) != saturation_length:
                raise ValueError(
                    f"{label} end_use saturations do not have the same length"
                )
            if len(getattr(i, "fuel_share")) != fuel_share_length:
                raise ValueError(
                    f"{label} end_use fuel_shares not have the same length"
                )
        return v

    @validator("end_uses")
    def validate_end_use_start_end_years(cls, v, values):
        label = values.get("label")
        start_year = [getattr(i, "start_year") for i in v]
        end_year = [getattr(i, "end_year") for i in v]
        if all(x != start_year[0] for x in start_year):
            raise ValueError(f"{label} end_use start_year values not equal")
        if all(y != end_year[0] for y in end_year):
            raise ValueError(f"{label} end_use end_year values not equal")
        return v

    @validator("start_year", always=True)
    def set_start_year(cls, v, values):
        if v is None:
            return getattr(values.get("end_uses")[0], "start_year")
        else:
            return v

    @validator("end_year", always=True)
    def set_end_year(cls, v, values):
        if v is None:
            return getattr(values.get("end_uses")[0], "end_year")
        else:
            return v

    _check_expected_list_length: classmethod = validator(
        "building_stock", allow_reuse=True
    )(check_expected_list_length)
