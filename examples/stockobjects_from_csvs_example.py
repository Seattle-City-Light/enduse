import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

from enduse.stockobjects import (
    Equipment,
    RampEfficiency,
    EndUse,
    Building,
    LoadShape,
)

from typing import List, Tuple, Dict

from enduse.stockturnover import BuildingModel


def create_filter_query(filters: List[Tuple[str, str]]) -> str:
    filter_str = " & ".join(
        [
            f"`{i[0]}` == '{i[1]}'" if isinstance(i[1], str) else f"`{i[0]}` == {i[1]}"
            for i in filters
        ]
    )
    return filter_str


def parse_equipment(
    start_year: int,
    end_year: int,
    end_use: str,
    equipment: str,
    level: int,
    subset: pd.DataFrame,
    shares: pd.DataFrame,
) -> Equipment:
    # create consumption
    cons = np.repeat(
        subset["Baseline Consumption (energy units / yr)"].to_list(),
        len(shares.columns),
    )
    # create useful like
    ul = np.repeat(subset["Effective Useful Life"].to_list(), len(shares.columns))
    # parse into Equipment object
    equipment = Equipment(
        equipment_label=equipment,
        efficiency_level=level,
        start_year=start_year,
        end_year=end_year,
        efficiency_share=shares.loc[end_use, level, equipment].values.tolist(),
        unit_consumption=cons.tolist(),
        useful_life=ul.tolist(),
    )
    return equipment


def parse_standards(
    end_use: str, standards: pd.DataFrame, equipment_dict: Dict[str, Equipment]
) -> RampEfficiency:
    standards = standards.loc[end_use].transpose().sort_index().drop_duplicates()
    ramp = RampEfficiency(
        ramp_label=f"{end_use} Standards",
        ramp_equipment=[equipment_dict[i] for i in standards.values],
        ramp_year=standards.index.get_level_values(0).to_list(),
    )
    return ramp


def parse_load_shape(
    end_use: str, load_shapes: pd.DataFrame, path: str, freq: str
) -> LoadShape:
    dim_filters = {
        "shape.type": "Load Shape",
        "in.geometry_building_type_recs": load_shapes["Building Type"].iloc[0],
        "in.puma": load_shapes["Puma"].iloc[0],
    }
    load_shape = LoadShape(
        source_file=path,
        dim_filters=dim_filters,
        value_filter=load_shapes["Load Shape"].iloc[0],
        freq=freq,
    )
    return load_shape


def parse_end_use(
    end_use: str,
    equipment: List[Equipment],
    ramp_efficiency: RampEfficiency,
    start_year: int,
    end_year: int,
    saturation: List[float],
    fuel_share: List[float],
    load_shape: LoadShape,
) -> EndUse:
    end_use = EndUse(
        end_use_label=end_use,
        equipment=equipment,
        start_year=start_year,
        end_year=end_year,
        ramp_efficiency=ramp_efficiency,
        saturation=saturation,
        fuel_share=fuel_share,
        load_shape=load_shape,
    )
    return end_use


def create_end_uses(
    segment: str,
    vintage: str,
    efficiency_share: pd.DataFrame,
    measure_inputs: pd.DataFrame,
    standards: pd.DataFrame,
    saturation: pd.DataFrame,
    fuel_share: pd.DataFrame,
    load_shape: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> Dict[str, Equipment]:

    year_range = np.arange(start_year, end_year + 1).astype("str")

    measure_vals = [
        "End Use",
        "Efficiency Level",
        "Efficiency Description",
        "Effective Useful Life",
        "Baseline Consumption (energy units / yr)",
    ]

    measure_index = ["End Use", "Efficiency Level", "Efficiency Description"]

    equip_filters = [("Segment", segment), ("Construction Vintage", vintage)]

    # get subset of equipment to model
    measures = (
        measure_inputs.query(create_filter_query(equip_filters))
        .filter(measure_vals)
        .drop_duplicates()
        .set_index(measure_index)
    )

    shares = (
        efficiency_share.query(create_filter_query(equip_filters))
        .set_index(measure_index)
        .filter(year_range)
    )

    segment_filt = [("Segment", segment)]

    standards = (
        standards.query(create_filter_query(segment_filt))
        .set_index(["End Use"])
        .filter(year_range)
    )

    saturation = (
        saturation.query(create_filter_query(equip_filters))
        .set_index(["End Use"])
        .filter(year_range)
    )

    fuel_share = (
        fuel_share.query(create_filter_query(equip_filters))
        .set_index(["End Use"])
        .filter(year_range)
    )

    load_shapes = load_shape.query(create_filter_query(equip_filters)).set_index(
        ["End Use"]
    )

    # loop over each end use
    end_uses = []
    for i, x in measures.groupby(level=0):
        # Parse Equipment
        equip_dict = OrderedDict()
        for (n, j), y in x.groupby(level=[1, 2]):
            equip_dict[j] = parse_equipment(
                start_year=start_year,
                end_year=end_year,
                end_use=i,
                equipment=j,
                level=n,
                subset=y,
                shares=shares,
            )

        # parse equipment standards
        ramp = parse_standards(
            end_use=i, standards=standards, equipment_dict=equip_dict
        )

        load_shape = parse_load_shape(
            end_use=i, load_shapes=load_shapes, path=netcdf_path, freq="H"
        )

        end_use = parse_end_use(
            end_use=i,
            equipment=list(equip_dict.values()),
            ramp_efficiency=ramp,
            start_year=start_year,
            end_year=end_year,
            saturation=saturation.loc[i].values.tolist(),
            fuel_share=fuel_share.loc[i].values.tolist(),
            load_shape=load_shape,
        )

        end_uses.append(end_use)

    return end_uses


# run model here
# set load shape path
csv_path = str((Path(__file__).parents[1] / "examples/inputs/residential/").as_posix())
netcdf_path = str(
    (Path(__file__).parents[1] / "outputs/loadshapes/resstock_loadshapes.nc").as_posix()
)

saturation = pd.read_csv(csv_path + "/saturations.csv")
fuel_share = pd.read_csv(csv_path + "/fuelSharesByVintage - Seattle Code Update.csv")
efficiency_share = pd.read_csv(csv_path + "/efficiencyShares.csv")
measure_inputs = pd.read_csv(csv_path + "/equipmentMeasureInputs.csv")
standards = pd.read_csv(csv_path + "/equipmentStandards.csv")
load_shapes = pd.read_csv(csv_path + "/loadShapes.csv")

# standards start in 2022 need to allign with forecast range which starts in 2021
standards["2021"] = standards["2022"]

standards_mapped = (
    measure_inputs[["Efficiency Description", "Efficiency Level"]]
    .drop_duplicates()
    .set_index("Efficiency Description")
    .to_dict()["Efficiency Level"]
)

end_use_params = {
    "segment": "Single Family",
    "vintage": "Existing",
    "efficiency_share": efficiency_share,
    "measure_inputs": measure_inputs,
    "standards": standards,
    "saturation": saturation,
    "load_shape": load_shapes,
    "fuel_share": fuel_share,
    "start_year": 2021,
    "end_year": 2041,
}

test_end_uses = create_end_uses(**end_use_params)

test_building = {
    "building_label": "Single Family",
    "end_uses": test_end_uses,
    "start_year": 2021,
    "end_year": 2041,
    "building_stock": np.linspace(200000, 200000, 21).tolist(),
    "segment": "Single Family",
    "construction_vintage": "Existing",
}

test_build_parsed = Building(**test_building)
test_stockturnover = BuildingModel(test_build_parsed)
