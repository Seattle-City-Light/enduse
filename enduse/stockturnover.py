import numpy as np
import pandas as pd
import xarray as xr

from enduse.stockobjects import Building, EndUse, RampEfficiency


def create_ramp_matrix(
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


def stock_turnover_calculation(
    equip_mat: np.ndarray, ul_mat: np.ndarray, ramp_mat: np.array
) -> np.ndarray:
    """
    Create an n x d matrix equal to dim(equip_mat)
    Stockturnover calculation is based on a expotential decay function
    Which is executed as an interative calcution based on
        stock_turnover[i, t + 1] = equip_mat[i, t] / ul_mat[i, t]
        where stock_turnover[i, t + 1] is based on ramp_mat[i, t]
    """

    # check for exogenous equipment additions and subtractions
    # from building_stock, saturaiton, fuel_share or efficiency_share
    prepend = np.reshape(equip_mat[:, 0], (equip_mat.shape[0], -1))
    equip_diff_mat = np.diff(equip_mat, prepend=prepend)
    equip_add_mat = np.cumsum(np.where(equip_diff_mat > 0, equip_diff_mat, 0), axis=1)
    equip_sub_mat = np.cumsum(np.where(equip_diff_mat < 0, equip_diff_mat, 0), axis=1)

    # container to hold stock turnover calculation
    # if no efficiency ramp or equipment is 1d vector then
    # populate with default equipment stock assumption
    equip_turn_cum_mat = np.copy(equip_mat)
    equip_turn_mat = np.zeros(equip_mat.shape)

    # stock turnover calc requires iteration over each forecast year
    # stock turnover calculation is an expotential decay function : E_t = E_0 * e^(-k * t)
    # but unable to vectorize since E_0 can change
    # and replaced equipment (E_0 - E_t) will also follow a decay function
    # possible TODO improve vectorization or use NUMBA/Cython if speed is a problem
    if not np.all(ramp_mat == 1):
        # normalize equipment counts for exogenous additions or subtractions
        # equip_turn_cum_mat = equip_mat - equip_add_mat + np.absolute(equip_sub_mat)

        # iterate over each forecast year
        for i in range(equip_mat.shape[1]):
            # stock turnover calc starts in first forecast year
            if i > 0:
                # equipment efficiency index
                ramp_loc = np.where(ramp_mat == 1)[0][i]

                # calculate equipment turnover for all equipment below minimum ramp level
                # this needs to reference equipment_turn_cum mat
                equip_turn = (
                    equip_turn_cum_mat[: ramp_loc + 1, i - 1]
                    / ul_mat[: ramp_loc + 1, i - 1]
                    * (1 - ramp_mat[: ramp_loc + 1, i])
                )

                # allocate turned over equipment to minumum ramp level
                equip_turn_mat[ramp_loc, i] = np.sum(equip_turn)

                # calculate total equipment for each efficiency level
                equip_turn_cum_mat[:, i] = (
                    # prior years value
                    equip_turn_cum_mat[:, i - 1]
                    # add stock converted to minimum efficiency share
                    + equip_turn_mat[:, i]
                    # subtract converted stock from original efficiency levels
                    - equip_turn
                    # account for any exogenous additions or substractions
                    + equip_diff_mat[:, i]
                )

    return equip_turn_cum_mat


def create_end_use_xarray(end_use: EndUse, building_stock: np.array) -> xr.Dataset:
    """Create arrays for stockturnover calc and load into xarray"""
    # 1d arrays
    bld_arr = np.array(building_stock)
    sat_arr = np.array(end_use.saturation)
    fs_arr = np.array(end_use.fuel_share)
    years_arr = np.arange(end_use.start_year, end_use.end_year + 1)
    level_arr = np.array([x.efficiency_level for x in end_use.equipment])
    label_arr = np.array([x.label for x in end_use.equipment])

    # 2d arrays
    eff_mat = np.array([np.array(x.efficiency_share) for x in end_use.equipment])
    con_mat = np.array([np.array(x.consumption) for x in end_use.equipment])
    ul_mat = np.array([np.array(x.useful_life) for x in end_use.equipment])

    # equipment stock calc
    equip_mat = bld_arr * sat_arr * fs_arr * eff_mat

    # handle efficiency ramp if it exists
    ramp_mat = np.ones(equip_mat.shape)
    if end_use.ramp_efficiency:
        ramp_mat = create_ramp_matrix(equip_mat, end_use.ramp_efficiency)

    st_mat = stock_turnover_calculation(equip_mat, ul_mat, ramp_mat)

    # TODO xarray supports no leap year method for datetimes
    # add noleap datetimes as a coordinate when load shapes are added
    data_xr = {
        "building_stock": (["years"], bld_arr),
        "saturation": (["years"], sat_arr),
        "fuel_share": (["years"], fs_arr),
        "ramp": (["efficeincy_level", "years"], ramp_mat),
        "efficiency_share": (["efficiency_level", "years"], eff_mat),
        "consumption": (["efficiency_level", "years"], con_mat),
        "useful_life": (["efficiency_level", "years"], ul_mat),
        "equipment_stock": (["efficiency_level", "years"], equip_mat),
        "st_mat": (["efficiency_level", "year"], st_mat),
    }

    coords_xr = {
        "efficiency_level": (level_arr),
        "years": (years_arr),
        "label": ("efficiency_share", label_arr),
    }

    end_use_xr = xr.Dataset(data_vars=data_xr, coords=coords_xr)

    return end_use_xr


def get_building_arrays(building: Building):

    end_uses = []
    for i in building.end_uses:
        end_uses.append(create_end_use_xarray(i, building.building_stock))

    return end_uses
