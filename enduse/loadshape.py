# need to create Equipment and RampEfficiency objects to test ramp
from matplotlib.style import use
from nbformat import ValidationError
import numpy as np
import pandas as pd
import xarray as xr
import dask

from typing import Optional, Tuple, Dict, List, Union
from glob import glob


def read_load_profiles_from_csvs(
    path: str, use_dask: bool = False
) -> Dict[str, pd.DataFrame]:
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
        df_dict = []
        for i in csv_paths:
            # append a tuple (label, dataframe)
            df_dict.append(
                (
                    i,
                    dask.delayed(
                        pd.read_csv(
                            i, parse_dates=["timestamp"], infer_datetime_format=True
                        )
                    ),
                )
            )
        df_dict = dict(dask.compute(*df_dict))
    else:
        df_dict = {
            i: pd.read_csv(i, parse_dates=["timestamp"], infer_datetime_format=True)
            for i in csv_paths
        }
    return df_dict


def load_shape_from_load_profile(
    label: str,
    load_profile: pd.DataFrame,
    timestamp_offset: Optional[pd.Timedelta] = None,
    meta_filt: List[str] = ["puma", "building_type"],
    values_filt: str = ["out.electricity"],
    freq: str = "H",
) -> Tuple[str, xr.Dataset]:
    """
    Generate xarray load shape dataset from NREL aggregate load profile
    Required params:
        load_profile: nrel load profile dataframe with parsed timestamp
        timestamp_offset: NREL default timestamp is in EST
            EX: for PST: pd.Timedelta(hours=-3, minutes=-15)
    """
    if timestamp_offset is None:
        timestamp_offset = pd.Timedelta(hours=0)

    load_profile_resampled = (
        load_profile.set_index(load_profile["timestamp"] + timestamp_offset)
        .filter(regex="|".join(values_filt))
        .resample(freq)
        .mean()
    )

    load_shape = (
        load_profile_resampled.div(load_profile_resampled.sum(axis=0))
        .fillna(0)
        .reset_index(drop=True)
    )

    load_shape.index.name = "hour_of_year"

    meta_cols = sorted(list(load_profile.filter(regex="|".join(meta_filt))))
    meta_fields = [np.unique(load_profile[i])[0] for i in meta_cols]

    for i, x in zip(meta_cols, meta_fields):
        load_shape[i] = x

    load_shape_xr = xr.Dataset.from_dataframe(
        load_shape.set_index(meta_cols, append=True)
    )
    return (label, load_shape_xr)


def load_shape_from_multiple_load_profiles(
    load_profiles: Tuple[str, pd.DataFrame],
    timestamp_offset: Optional[pd.Timedelta] = None,
    meta_filt: List[str] = ["puma", "building_type"],
    values_filt: List[str] = ["out.electricity"],
    freq: str = "H",
    concat: bool = True,
    use_dask: bool = False,
) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
    """
    Wrapper function to handle multiple load profiles returned from read_profiles_from_csvs
    Requierd params:
    ...
    """
    xarrays = []
    if use_dask:
        for i, x in load_profiles.items():
            xarrays.append(
                dask.delayed(load_shape_from_load_profile)(
                    i, x, timestamp_offset, meta_filt, values_filt, freq
                )
            )
        xarrays = dask.compute(*xarrays)
    else:
        for i, x in load_profiles.items():
            xarrays.append(
                load_shape_from_load_profile(
                    i, x, timestamp_offset, meta_filt, values_filt, freq
                )
            )
    xarrays = dict(xarrays)
    # merge into single xarray
    if concat:
        # need to check that coords are the same in each xarray
        coords = [list(i.coords) for i in xarrays.values()]
        if all(i == coords[0] for i in coords):
            xarrays = xr.merge(list(xarrays.values()))
        else:
            raise ValidationError("Load profiles have different meta data label names")
    return xarrays


class LoadShapesFromLoadProfiles:
    def __init__(
        self,
        load_profiles: Optional[Dict[str, pd.DataFrame]] = None,
        timestamp_offset: Optional[pd.Timedelta] = None,
        meta_filt: List[str] = ["puma", "building_type"],
        values_filt: List[str] = ["out.electricity"],
        freq: str = "H",
        use_dask: bool = False,
        dir: Optional[str] = None,
    ):
        # initialize params
        # don't initialize load_profiles as attr to limit memory impacts
        self.timestamp_offset = timestamp_offset
        self.meta_filt = meta_filt
        self.values_filt = values_filt
        self.freq = freq
        self.use_dask = use_dask

        self._validate_load_profile_params(load_profiles, dir)

        if dir:
            self._load_shapes = self._parse_load_shapes(
                read_load_profiles_from_csvs(dir)
            )
        else:
            self._load_shapes = self._parse_load_shapes(load_profiles)

    @property
    def load_shapes(self):
        return self._load_shapes

    def _validate_load_profile_params(self, load_profiles, dir):
        if (load_profiles is None and dir is None) or (
            load_profiles is not None and dir is not None
        ):
            raise ValidationError(
                "Must provide either load_profiles: Dict[str, pd.DataFrame] or dir: str"
            )

    def _parse_load_shapes(
        self, load_profiles: Dict[str, pd.DataFrame],
    ) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        load_shapes = load_shape_from_multiple_load_profiles(
            load_profiles=load_profiles,
            timestamp_offset=self.timestamp_offset,
            meta_filt=self.meta_filt,
            values_filt=self.values_filt,
            freq=self.freq,
            use_dask=self.use_dask,
        )
        return load_shapes

    def to_netcdf(self, path: str) -> None:
        self.load_shapes.to_netcdf(path)
