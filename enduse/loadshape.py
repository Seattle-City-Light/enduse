# need to create Equipment and RampEfficiency objects to test ramp
import numpy as np
import pandas as pd
import xarray as xr
import dask

from typing import Optional, List

# from glob import glob


def read_load_profiles_from_csvs(
    path: str, use_dask: bool = False
) -> List[pd.DataFrame]:
    """
    Read in a single NREL load profile .csv
    Note that dask will only provide a performance boost when there are many large .csv files
    Since dask requires some overheard for scheduling
    Required params:
        path: path/to/dir/
        use_dask: use dask to parallelize operations (default False)
    """
    csv_paths = glob(f"{path}/*.csv")
    if use_dask:
        dfs = []
        for i in csv_paths:
            dfs.append(
                dask.delayed(
                    pd.read_csv(
                        i, parse_dates=["timestamp"], infer_datetime_format=True
                    )
                )
            )
    else:
        dfs = [pd.read_csv(i) for i in csv_paths]
    return dfs


def load_shape_xarray_from_nrel_load_profile(
    nrel_load_profile: pd.DataFrame,
    timestamp_offset: Optional[pd.Timedelta] = None,
    nrel_data_vars_filt: str = "out.electricity",
    freq: str = "H",
) -> xr.Dataset:
    """
    Generate xarray load shape dataset from NREL aggregate load profile
    Required params:
        ...
    """

    load_profile_resampled = (
        nrel_load_profile.set_index(nrel_load_profile["timestamp"] - timestamp_offset)
        .filter(like=nrel_data_vars_filt)
        .resample(freq)
        .mean()
    )

    load_shape = (
        load_profile_resampled.div(load_profile_resampled.sum(axis=0))
        .fillna(0)
        .reset_index(drop=True)
    )

    load_shape.index.name = "hour_of_year"

    meta_cols = list(nrel_load_profile.filter(like="in."))
    meta_fields = list(np.unique(nrel_load_profile.filter(like="in.").values))

    load_shape_xr = xr.Dataset.from_dataframe(load_shape).assign_coords(
        {i: x for i, x in zip(meta_cols, meta_fields)}
    )

    return load_shape_xr


def test():
    return None
