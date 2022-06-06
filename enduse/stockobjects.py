import numpy as np

from typing import List, Optional, Dict, Union
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

# TODO modify ramp to handle multiple efficiency ramp levels
# TODO transition matrix could be used to handle multiple efficiency levels
# TODO modify stock objects to handle different adoption logic: type='decay', type='logit'


def check_expected_list_length(v: list, values: dict):
    label = values["label"]
    exp_len = values["end_year"] - values["start_year"] + 1
    if len(v) != exp_len:
        raise ValueError(f"{label} has list len {len(v)} but expected len {exp_len}")
    return v


class LoadShape(BaseModel):
    source_file: StrictStr
    dim_filters: Dict[StrictStr, StrictStr]
    value_filter: StrictStr
    freq: StrictStr = "H"
    extra_dims: Optional[Dict[StrictStr, int]]

    @validator("source_file")
    def validate_source_file(cls, v):
        if not Path(v).is_file():
            raise FileNotFoundError(f"{v} is invalid path")
        return v

    @validator("freq")
    def valdidate_freq(cls, v):
        if v not in ["H", "D"]:
            raise ValueError(f"{v} invalid frequency not in: 'H' or 'D'")
        return v

    @validator("extra_dims")
    def validate_load_shape_dims_len(cls, v):
        if len(v.keys()) > 2:
            raise ValueError(
                f"Provided load shape has extra_dims = {len(v.keys())} and exceeds max allowed (2)"
            )
        return v

    @validator("extra_dims")
    def validate_load_shape_dims_keys(cls, v):
        for i in v.keys():
            if i not in ["weather_year", "forecast_year"]:
                raise ValueError(
                    "f {i} invalid dim name must be in: 'weather_year' or 'forecast_year"
                )
        return v


class Equipment(BaseModel):
    label: StrictStr = Field(None, alias="equipment_label")
    efficiency_level: PositiveInt
    start_year: PositiveInt
    end_year: PositiveInt
    efficiency_share: List[confloat(ge=0, le=1)]
    unit_consumption: List[confloat(ge=0)]
    useful_life: List[PositiveInt]
    load_shape: Optional[LoadShape]
    end_use_override: Optional[str]

    # check list lengths match expected
    _check_expected_list_length: classmethod = validator(
        "efficiency_share", "unit_consumption", "useful_life", allow_reuse=True
    )(check_expected_list_length)


class RampEfficiency(BaseModel):
    """
    Valid ramp_logic: [fixed, forced]
        fixed: prioritize equipment allocation based on exogenous changes in building, end_use and equipment (existing construction)
        forced: prioritize equipment allocation based on ramp efficiency (new construciton)
    """

    label: StrictStr = Field(None, alias="ramp_label")
    ramp_equipment: List[Equipment]
    ramp_year: Optional[List[PositiveInt]]
    ramp_logic: Optional[str] = "exog"

    # inherit ramp_year if not provided
    @validator("ramp_year", pre=True, always=True)
    def set_ramp_year(cls, v, values):
        if v is None:
            v = [values["ramp_equipment"][0].start_year]
        return v

    # check that ramp year falls within equipment year range
    @validator("ramp_year")
    def validate_ramp_year_in_range(cls, v, values):
        label = values["label"]
        for i, x in zip(v, values["ramp_equipment"]):
            if i < x.start_year or i > x.end_year:
                raise ValueError(f"{label} ramp_year outside of equipment year range")
        return v

    # if ramp year is provided check that first ramp equals first year in equipment
    @validator("ramp_year")
    def validate_first_ramp_year(cls, v, values):
        label = values["label"]
        exp_min_ramp_year = values["ramp_equipment"][0].start_year
        if min(v) > exp_min_ramp_year:
            raise ValueError(
                f"{label} has min ramp_year of {v} but expected {exp_min_ramp_year}"
            )
        return v

    # validate that multiple ramp years are sequential
    @validator("ramp_year")
    def validate_sequential_ramp_year(cls, v, values):
        label = values["label"]
        if len(v) > 1:
            for i, x in enumerate(v):
                if i > 0:
                    if v[i] <= v[i - 1]:
                        raise ValueError(f"{label} invalid ramp_year order")
        return v

    @validator("ramp_logic")
    def validate_ramp_logic(cls, v):
        if v not in ["exog", "forced"]:
            raise ValueError(
                f"{v} is invalid ramp_type. Valid ramp_types: [exog, forced]"
            )

    # check that ramp inputs are a consistent length
    @root_validator(skip_on_failure=True)
    def validate_same_list_length(cls, values):
        label = values["label"]
        length = len(values["ramp_year"])
        list_fields = ["ramp_year", "ramp_equipment"]
        if any(len(values[i]) != length for i in list_fields):
            raise ValueError(f"{label} list fields do not have the same length")
        return values


class EndUse(BaseModel):
    label: StrictStr = Field(None, alias="end_use_label")
    equipment: List[Equipment]
    ramp_efficiency: Optional[RampEfficiency]
    start_year: PositiveInt
    end_year: PositiveInt
    saturation: List[confloat(ge=0)]
    fuel_share: List[confloat(ge=0, le=1)]
    load_shape: Optional[LoadShape]

    _has_equipment_load_shape = PrivateAttr()
    _has_end_use_override = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._has_equipment_load_shape = any(
            [isinstance(i.load_shape, LoadShape) for i in self.equipment]
        )
        self._has_end_use_override = any([i.end_use_override for i in self.equipment])

    # check that all equipment has same list length
    @validator("equipment")
    def validate_equipment_list_length(cls, v, values):
        label = values["label"]
        length = len(v[0].efficiency_share)
        for i in v:
            if len(i.efficiency_share) != length:
                raise ValueError(
                    f"{label} equipment efficiency_share do not have the same length"
                )
        return v

    # check that equipment allocation each yeah sums to 100%
    @validator("equipment")
    def validate_equipment_allocation(cls, v, values):
        label = values["label"]
        allocations = np.array([i.efficiency_share for i in v])
        if np.any(np.round(allocations.sum(axis=0), 0) != 1):
            raise ValueError(
                f"{label} equipment efficiency_share allocations do not sum to 1"
            )
        return v

    # check that equipment has sequential efficiency levels
    @validator("equipment")
    def validate_equipment_efficeincy_levels(cls, v, values):
        label = values["label"]
        levels = [i.efficiency_level for i in v]
        expected_levels = [*range(min(levels), max(levels) + 1)]
        if levels != expected_levels:
            raise ValueError(f"{label} equipment efficiency_levels are not sequential")
        return v

    # check equipment has consistent start and end years
    @validator("equipment")
    def validate_equipment_start_end_years(cls, v, values):
        label = values["label"]
        start_year = [i.start_year for i in v]
        end_year = [i.end_year for i in v]
        if all(x != start_year[0] for x in start_year):
            raise ValueError(f"{label} equipment start_year values not equal")
        if all(y != end_year[0] for y in end_year):
            raise ValueError(f"{label} equipment end_year values not equal")
        return v

    @validator("ramp_efficiency")
    def check_ramp_equipment_valid(cls, v, values):
        if "equipment" in values.keys():
            label = values["label"]
            ramp_labels = [i.label for i in v.ramp_equipment]
            equipment_labels = [i.label for i in values["equipment"]]
            if not any([i in equipment_labels for i in ramp_labels]):
                raise ValueError(f"{label} ramp_equipment not in equipment")
        return v

    # TODO add validator for equipment load_shape length

    _check_expected_list_length: classmethod = validator(
        "saturation", "fuel_share", allow_reuse=True
    )(check_expected_list_length)


class Building(BaseModel):
    label: StrictStr = Field(None, alias="building_label")
    end_uses: List[EndUse]
    start_year: PositiveInt
    end_year: PositiveInt
    building_stock: List[PositiveFloat]
    segment: Optional[StrictStr]
    construction_vintage: Optional[StrictStr]

    # private attribute to track max # of efficiency shares and load shapes dims
    # need to ensure all numpy arrays in enduse -> stockturnover have same dims
    # xarray requires Datasets to have same dims
    _end_use_len: int = PrivateAttr()
    _has_end_use_load_shape: bool = PrivateAttr()
    _load_shape_freq: Union[None, StrictStr] = PrivateAttr()
    _load_shape_extra_dims: Union[None, dict] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._end_use_len = max([len(i.equipment) for i in self.end_uses])
        self._has_end_use_load_shape = any(
            [isinstance(i.load_shape, LoadShape) for i in self.end_uses]
        )

        # if load shapes are provided then get max dimensions
        if self._has_end_use_load_shape:
            self._load_shape_freq = self._get_load_shape_freq()
            if self._has_load_shape_extra_dims():
                self._load_shape_extra_dims = self._get_load_shape_extra_dims()
        else:
            self._load_shape_freq = None

    @validator("end_uses")
    def validate_end_use_list_length(cls, v, values):
        label = values["label"]
        saturation_length = len(v[0].saturation)
        fuel_share_length = len(v[0].fuel_share)
        for i in v:
            if len(i.saturation) != saturation_length:
                raise ValueError(
                    f"{label} end_use saturations do not have the same length"
                )
            if len(i.fuel_share) != fuel_share_length:
                raise ValueError(
                    f"{label} end_use fuel_shares not have the same length"
                )
        return v

    @validator("end_uses")
    def validate_end_use_start_end_years(cls, v, values):
        label = values["label"]
        start_year = [i.start_year for i in v]
        end_year = [i.end_year for i in v]
        if all(x != start_year[0] for x in start_year):
            raise ValueError(f"{label} end_use start_year values not equal")
        if all(y != end_year[0] for y in end_year):
            raise ValueError(f"{label} end_use end_year values not equal")
        return v

    @validator("end_uses")
    def validate_end_use_load_shape_interval(cls, v):
        freqs = [i.load_shape.freq for i in v if i.load_shape]
        if len(set(freqs)) > 1:
            raise ValueError(f"Load shape frequencies are inconsistent")
        return v

    _check_expected_list_length: classmethod = validator(
        "building_stock", allow_reuse=True
    )(check_expected_list_length)

    def _get_load_shape_freq(self):
        """Extract frequency from load shape"""
        freq = set([i.load_shape.freq for i in self.end_uses if i.load_shape])
        return list(freq)[0]

    def _has_load_shape_extra_dims(self):
        """Check if extra dims provided in load shapes"""
        extra_dims = [i.load_shape.extra_dims for i in self.end_uses if i.load_shape]
        return any(extra_dims)

    def _get_load_shape_extra_dims(self):
        """Get extra dims from end_uses"""
        extra_dims = []
        for i in self.end_uses:
            if i.load_shape is not None:
                if i.load_shape.extra_dims is not None:
                    extra_dims.append(len(i.load_shape.extra_dims.keys()))
                else:
                    extra_dims.append(0)
            else:
                extra_dims.append(0)

        max_extra_dims = sorted(
            self.end_uses[
                extra_dims.index(max(extra_dims))
            ].load_shape.extra_dims.items()
        )
        return max_extra_dims

