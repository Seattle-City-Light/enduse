import os
import numpy as np
import pandas as pd
import xarray as xr

from typing import List, Dict, Optional
from datetime import datetime

from enduse.stockobjects import Building, EndUse, RampEfficiency


def _create_ramp_matrix(
    equip_mat: np.ndarray, ramp_efficiency: RampEfficiency
) -> np.array:
    """
    Create an n x d matrix equal to dim(equip_mat) 
    Populate with 0-1 to identify ramp logic for turnover calculation
    
    For 3 types of equiment [1, 2, 3] and forecast periods [1, 2, 3, 4] 
    where equipment 2 sets ramp logic in periods [1, 2] and
    equipment 3 sets ramp logic in periods [3, 4]
    returned matrix would be 3 x 4: 
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    """
    ramp_equipment = ramp_efficiency.ramp_equipment
    ramp_year = ramp_efficiency.ramp_year
    # create efficiency ramp array
    # ramp logic depends on number of equipment ramp levels
    ramp_mat = np.zeros(equip_mat.shape)
    offset = 0
    for i, x in enumerate(ramp_equipment):
        ramp_level = x.efficiency_level
        # if only 1 efficiency ramp then ramp is fixed for entire forecast period
        if len(ramp_equipment) == 1:
            ramp_mat[ramp_level - 1, :] = 1
        # if multiple efficiency ramps then interpolate as a step type function based on ramp_years
        else:
            if i < len(ramp_equipment) - 1:
                ramp_mat[
                    ramp_level - 1 :,
                    offset : offset + (ramp_year[i + 1] - ramp_year[i]),
                ] = 1
                # update offset counter
                offset = np.nonzero(ramp_mat)[1][-1] + 1
            else:
                ramp_mat[
                    ramp_level - 1 :, offset : offset + (x.end_year - ramp_year[i] + 1)
                ] = 1
    return ramp_mat


def _create_stock_turnover(
    equip_mat: np.ndarray, ul_mat: np.ndarray, ramp_mat: np.array
) -> np.ndarray:
    """
    Create an n x d matrix equal to dim(equip_mat)
    Stockturnover calculation is based on a expotential decay function
    Which is executed as an iterative calcution based on
        stock_turnover[i, t + 1] = equip_mat[i, t] / ul_mat[i, t]
        where stock_turnover[i, t + 1] is based on ramp_mat[i, t]
    """

    # check for exogenous equipment additions and subtractions
    # from building_stock, saturaiton, fuel_share or efficiency_share
    prepend = np.reshape(equip_mat[:, 0], (equip_mat.shape[0], -1))
    equip_diff_mat = np.diff(equip_mat, prepend=prepend)

    # container to hold stock turnover calculation
    # if no efficiency ramp or equipment is 1d vector then
    # populate with default equipment stock assumption
    equip_turn_cum_mat = np.copy(equip_mat)
    equip_turn_mat = np.zeros(equip_mat.shape)

    # stock turnover calc requires iteration over each forecast year
    # stock turnover calculation is an expotential decay function : E_t = E_0 * e^(-k * t)
    # but unable to vectorize since E_0 can change
    # and replaced equipment (E_0 - E_t) will also follow a decay function
    # possible TODO improve vectorization or use Numba/Cython if speed is a problem
    if not np.all(ramp_mat == 1):
        for i in range(equip_mat.shape[1]):
            if i > 0:
                # identify minimum ramp efficiency index
                ramp_loc = np.where(ramp_mat == 1)[0][i]
                # calculate equipment turnover for all equipment below minimum ramp level
                equip_turn = np.zeros(equip_mat.shape[0])
                # this needs to reference equipment_turn_cum mat
                equip_turn[: ramp_loc + 1] = (
                    equip_turn_cum_mat[: ramp_loc + 1, i - 1]
                    / ul_mat[: ramp_loc + 1, i - 1]
                    * (1 - ramp_mat[: ramp_loc + 1, i])
                )
                # allocate turned over equipment to minumum ramp level
                equip_turn_mat[ramp_loc, i] = np.sum(equip_turn)
                # calculate total equipment for each efficiency level
                equip_turn_cum_mat[:, i] = (
                    # prior years total equipment stock
                    equip_turn_cum_mat[:, i - 1]
                    # add equipment stock converted to minimum efficiency share
                    + equip_turn_mat[:, i]
                    # subtract converted equipment stock from original efficiency levels
                    - equip_turn
                    # account for any exogenous equiment additions or substractions
                    + equip_diff_mat[:, i]
                )

    return equip_turn_cum_mat


def _create_xarray_from_end_use(building: Building, end_use: EndUse) -> xr.Dataset:
    """
    Extract params from EndUse object and convert to numpy arrarys
    pass numpy arrays to stock turnover calc
    return xarray dataset with detailed stock turnover calculation details
    """
    # 1d arrays
    bld_arr = np.array(building.building_stock)
    sat_arr = np.array(end_use.saturation)
    fs_arr = np.array(end_use.fuel_share)
    level_arr = np.array([x.efficiency_level for x in end_use.equipment])
    el_label_arr = np.array([x.label for x in end_use.equipment])

    # 2d arrays
    eff_mat = np.array([np.array(x.efficiency_share) for x in end_use.equipment])
    con_mat = np.array([np.array(x.consumption) for x in end_use.equipment])
    ul_mat = np.array([np.array(x.useful_life) for x in end_use.equipment])

    # equipment stock calc
    equip_mat = bld_arr * sat_arr * fs_arr * eff_mat

    # handle efficiency ramp if it exists
    ramp_mat = np.ones(equip_mat.shape)
    if end_use.ramp_efficiency:
        ramp_mat = _create_ramp_matrix(equip_mat, end_use.ramp_efficiency)

    st_mat = _create_stock_turnover(equip_mat, ul_mat, ramp_mat)

    # build the xarray
    data_xr = {
        "building_stock": (["year"], bld_arr),
        "saturation": (["year"], sat_arr),
        "fuel_share": (["year"], fs_arr),
        "ramp_matrix": (["efficiency_level", "year"], ramp_mat),
        "efficiency_share": (["efficiency_level", "year"], eff_mat),
        "unit_consumption": (["efficiency_level", "year"], con_mat),
        "useful_life": (["efficiency_level", "year"], ul_mat),
        "init_equipment_stock": (["efficiency_level", "year"], equip_mat),
        "equipment_stock": (["efficiency_level", "year"], st_mat),
        "consumption": (["efficiency_level", "year"], st_mat * con_mat),
    }

    coors_xr = {
        "efficiency_level": level_arr,
        "efficiency_label": ("efficiency_level", el_label_arr),
        "year": np.arange(end_use.start_year, end_use.end_year + 1),
        "end_use_label": end_use.label,
        "building_label": building.label,
    }

    attrs_xr = {"datetime_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # create xarray dataset and check if redim required
    # xarray requires equal dims for netcdf export
    dataset_xr = xr.Dataset(data_vars=data_xr, coords=coors_xr, attrs=attrs_xr)
    if dataset_xr.dims["efficiency_level"] < building._end_use_len:
        dataset_xr = dataset_xr.reindex(
            {"efficiency_level": np.arange(1, building._end_use_len + 1)}
        )
    return dataset_xr


def _create_xarray_list_from_building(building: Building) -> Dict[str, xr.Dataset]:
    """Dispatcher to create xarray DataSet from building end-use objects"""
    xr_datasets = {}
    for i in building.end_uses:
        xr_datasets[i.label] = _create_xarray_from_end_use(building, i)
    return xr_datasets


class BuildingModel:
    def __init__(self, building: Building):
        self.building_label = building.label
        self.model = self._create_model(building)

    def _create_model(self, building: Building) -> xr.DataArray:
        return _create_xarray_list_from_building(building)

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate list of xarray datasets into a single dataframe"""
        # xarray datasets may contain nans to keep consistent dims
        # need to drop nan rows before converting to pd.DataFrame
        dataframe = pd.concat([i.to_dataframe() for i in self.model.values()])
        return dataframe.dropna(axis=0).reset_index()

    def to_netcdf(self, path: str, file_name: Optional[str] = None) -> None:
        """
        Export xarrays to netcdf file structure
        Required params:
            path: folder path ex: "./some_dir/"
        Optional params:
            file_name: overrides default file name (not recommended)
        """
        # build file path
        if file_name is not None:
            path += file_name
        else:
            path += self.building_label + ".nc"

        # if file alread exists then remove
        if os.path.exists(path):
            os.remove(path)

        # will append xarray datasets to file
        # file has must exist for append mode to work
        for n, (i, x) in enumerate(self.model.items()):
            # write first file without append
            if n == 0:
                x.to_netcdf(path, group=i)
            else:
                x.to_netcdf(path, group=i, mode="a")
