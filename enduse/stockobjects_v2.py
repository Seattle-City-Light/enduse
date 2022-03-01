import json
import numpy as np

from typing import List, Optional
from pydantic import (
    BaseModel,
    StrictStr,
    PositiveInt,
    PositiveFloat,
    confloat,
    Field,
    validator,
    root_validator,
)

# TODO alias on all field names for DF conversion?
# TODO validate childred/parent list lengths
# TODO move protype code to tests


def check_expected_list_length(v: list, values: dict) -> list:
    label = values.get("label")
    exp_len = values.get("end_year") - values.get("start_year") + 1
    if len(v) != exp_len:
        raise ValueError(f"{label} has list len {len(v)} but expected len {exp_len}")
    return v


class Equipment(BaseModel):
    label: StrictStr = Field(None, alias="equipment_label")
    efficiency_level: PositiveInt
    start_year: PositiveInt
    end_year: PositiveInt
    efficiency_share: List[confloat(ge=0, le=1)]
    consumption: List[PositiveFloat]
    useful_life: List[PositiveInt]

    # check efficiency share matches start and end years
    _check_expected_list_length: classmethod = validator(
        "efficiency_share", "consumption", "useful_life", allow_reuse=True
    )(check_expected_list_length)


class EfficiencyRamp(BaseModel):
    label: StrictStr = Field(None, alias="ramp_label")
    ramp_start_year: List[PositiveInt]
    ramp_end_year: List[PositiveInt]
    ramp_equipment: List[Equipment]

    # check that ramp inputs are a consistent length
    @root_validator
    def validate_same_list_length(cls, values):
        label = values.get("label")
        length = len(values.get("ramp_start_year"))
        list_fields = ["ramp_start_year", "ramp_end_year", "ramp_equipment"]
        if any(len(values[i]) != length for i in list_fields):
            raise ValueError(f"{label} list fields do not have the same length")
        return values


class EndUse(BaseModel):
    label: StrictStr = Field(None, alias="end_use_label")
    equipment: List[Equipment]
    efficiency_ramp: List[EfficiencyRamp]
    start_year: Optional[PositiveInt] = None
    end_year: Optional[PositiveInt] = None
    saturation: List[PositiveFloat]
    fuel_share: List[confloat(ge=0, le=1)]

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

    @validator("equipment")
    def validate_equipment_start_end_years(cls, v, values):
        label = values.get("label")
        start_year = [getattr(i, "start_year") for i in v]
        end_year = [getattr(i, "end_year") for i in v]
        if all(x != start_year[0] for x in start_year):
            if all(y != end_year[0] for y in end_year):
                raise ValueError(
                    f"{label} equipment start_year and end_year values not equal"
                )
        return v

    @validator("start_year", always=True)
    def set_start_year(cls, v, values) -> int:
        if v is None:
            return getattr(values.get("equipment")[0], "start_year")
        else:
            return v

    @validator("end_year", always=True)
    def set_end_year(cls, v, values) -> int:
        if v is None:
            return getattr(values.get("equipment")[0], "end_year")
        else:
            return v

    _check_expected_list_length: classmethod = validator(
        "saturation", "fuel_share", allow_reuse=True
    )(check_expected_list_length)


class Building(BaseModel):
    label: StrictStr = Field(None, alias="building_label")
    building_stock: List[PositiveFloat]
    end_uses: List[EndUse]
    customer_class: Optional[StrictStr] = None
    construction_vintage: Optional[StrictStr] = None


equipment = []

equipment.append(
    {
        "equipment_label": "Below Standard Heat Pump - SEER/EER 10/9.2 and HSPF 7.2 (Split System)",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2031,
        "efficiency_share": np.linspace(0.75, 0.75, 10).tolist(),
        "consumption": np.linspace(8742, 8742, 10).tolist(),
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

equipment_parsed = [Equipment(**i) for i in equipment]

ramp = []

ramp.append(
    {
        "ramp_start_year": [2022, 2025],
        "ramp_end_year": [2024, 2031],
        "ramp_equipment": [equipment_parsed[1], equipment_parsed[2]],
    }
)

ramp_parsed = [EfficiencyRamp(**i) for i in ramp]

end_use = []

end_use.append(
    {
        "end_use_label": "Heat Pump",
        "saturation": np.linspace(0.25, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": equipment_parsed,
        "efficiency_ramp": ramp_parsed,
    }
)

end_use_parsed = EndUse(**end_use[0])

# building = []

# building.append(
#     {
#         "customer_class": "Residential",
#         "segment": "Single Family",
#         "construction_vintage": "Existing",
#         "start_year": 2022,
#         "end_year": 2031,
#         "building_stock": np.linspace(1000, 1000, 10).tolist(),
#         "end_uses": end_use,
#     }
# )
