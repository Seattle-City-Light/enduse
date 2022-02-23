import json
import numpy as np

from typing import List, Optional
from pydantic import (
    BaseModel,
    StrictStr,
    StrictInt,
    StrictFloat,
    Field,
    validator,
    root_validator,
)

# TODO alias on all field names for DF conversion?
# TODO validate > 0 list fields
# validate buildings


def check_expected_list_length(v: list, values: dict) -> list:
    label = values.get("label")
    exp_len = values.get("end_year") - values.get("start_year") + 1
    if len(v) != exp_len:
        raise ValueError(f"{label} has list len {len(v)} but expected len {exp_len}")
    return v


def check_values_between_zero_one(v: list, values: dict) -> list:
    label = values.get("label")
    if not ((np.array(v) >= 0) & (np.array(v) <= 1)).all():
        raise ValueError(f"{label} has value outside 0 and 1")
    return v


class Equipment(BaseModel):
    label: StrictStr = Field(None, alias="equipment_label")
    efficiency_level: StrictInt
    start_year: StrictInt
    end_year: StrictInt
    efficiency_share: List[StrictFloat]
    consumption: List[StrictFloat]
    useful_life: List[StrictFloat]

    # validators
    _check_expected_list_length: classmethod = validator(
        "efficiency_share", "consumption", "useful_life", allow_reuse=True
    )(check_expected_list_length)

    _check_values_between_zero_one: classmethod = validator("efficiency_share")(
        check_values_between_zero_one
    )


class EfficiencyRamp(BaseModel):
    label: StrictStr = Field(None, alias="ramp_label")
    ramp_start_year: List[StrictInt]
    ramp_end_year: List[StrictInt]
    ramp_equipment: List[Equipment]

    @root_validator
    def validate_same_list_length(cls, values):
        length = len(values.get("ramp_start_year"))
        list_fields = ["ramp_start_year", "ramp_end_year", "ramp_equipment"]
        if any(len(values[i]) != length for i in list_fields):
            raise ValueError("List fields do not have the same length")
        return values


class EndUse(BaseModel):
    label: StrictStr = Field(None, alias="end_use_label")
    start_year: StrictInt
    end_year: StrictInt
    saturation: List[StrictFloat]
    fuel_share: List[StrictFloat]
    equipment: List[Equipment]
    efficiency_ramp: List[EfficiencyRamp]

    _check_expected_list_length: classmethod = validator(
        "saturation", "fuel_share", allow_reuse=True
    )(check_expected_list_length)

    _check_values_between_zero_one: classmethod = validator("fuel_share")(
        check_values_between_zero_one
    )

    @validator("equipment")
    def validate_equipment_allocation(cls, v, values):
        label = values.get("_label")
        allocations = np.array([getattr(i, "efficiency_share") for i in v])
        if np.any(allocations.sum(axis=0) != 1):
            raise ValueError(
                f"{label} equipment efficiency_share allocations do not sum to 1"
            )


class Building(BaseModel):
    label: StrictStr = Field(None, alias="building_label")
    start_year: StrictInt
    end_year: StrictInt
    building_stock: List[StrictFloat]
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

equipment_parsed = {i: Equipment(**x) for i, x in enumerate(equipment)}

ramp = []

ramp.append(
    {
        "ramp_start_year": [2022, 2025],
        "ramp_end_year": [2024, 2031],
        "ramp_equipment": [equipment_parsed[1], equipment_parsed[2]],
    }
)

ramp_parsed = EfficiencyRamp(**ramp[0])

end_use = []

end_use.append(
    {
        "end_use_label": "Heat Pump",
        "start_year": 2022,
        "end_year": 2031,
        "saturation": np.linspace(0.25, 0.25, 10).tolist(),
        "fuel_share": np.linspace(1, 1, 10).tolist(),
        "equipment": list(equipment_parsed.values()),
        "efficiency_ramp": ramp,
    }
)

building = []

building.append(
    {
        "customer_class": "Residential",
        "segment": "Single Family",
        "construction_vintage": "Existing",
        "start_year": 2022,
        "end_year": 2031,
        "building_stock": np.linspace(1000, 1000, 10).tolist(),
        "end_uses": end_use,
    }
)
