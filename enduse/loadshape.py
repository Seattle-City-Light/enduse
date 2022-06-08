# need to create Equipment and RampEfficiency objects to test ramp
from nbformat import ValidationError
import numpy as np
import pandas as pd
import xarray as xr
import dask

from typing import Optional, Tuple, Dict, List, Union
from glob import glob


freq_dict = {"H": (8760, "hour_of_year"), "D": (365, "day_of_year")}


def read_load_profiles_from_csvs(path: str) -> Dict[str, pd.DataFrame]:
    """
    Read in single or multiple NREL load profile .csv(s)
    Required params:
        path: path/to/dir/
    """
    csv_paths = glob(f"{path}/*.csv")
    df_dict = {
        i: pd.read_csv(i, parse_dates=["timestamp"], infer_datetime_format=True)
        for i in csv_paths
    }
    return df_dict


def xarray_from_load_profiles() -> None:
    return None


def load_shape_from_load_profile(
    label: str,
    load_profile: pd.DataFrame,
    timestamp_offset: Optional[pd.Timedelta] = None,
    meta_filt: List[str] = ["puma", "building_type"],
    values_filt: List[str] = ["out.electricity"],
    freq: str = "H",
    attach_temp: bool = True,
    agg_cols: Optional[List[Dict[str, Union[list, str]]]] = None,
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

    # need to create timestamp from a resample to avoid conflicts with aggregation steps below
    timestamp = (
        load_profile.set_index(load_profile["timestamp"] + timestamp_offset)
        .resample(freq)
        .sum()
        .index
    )

    if agg_cols:
        for i in agg_cols:
            load_profile[i["label"]] = load_profile.filter(
                regex="|".join(i["agg_cols"])
            ).agg(i["agg_func"], axis=1)

            values_filt = values_filt + [i["label"]]

    load_profile_resampled = (
        load_profile.set_index(load_profile["timestamp"] + timestamp_offset)
        .filter(regex="|".join(values_filt))
        .resample(freq)
        .mean()
        .reset_index(drop=True)
    )

    load_shape = (
        load_profile_resampled.div(load_profile_resampled.sum(axis=0))
        .fillna(0)
        .reset_index(drop=True)
    )

    for i in [load_profile_resampled, load_shape]:
        i.index.name = freq_dict[freq][1]

    meta_cols = sorted(list(load_profile.filter(regex="|".join(meta_filt))))
    meta_fields = [np.unique(load_profile[i])[0] for i in meta_cols]

    for i, x in zip(meta_cols, meta_fields):
        load_profile_resampled[i] = x
        load_shape[i] = x

    if attach_temp:
        temp_resampled = (
            load_profile.set_index(load_profile["timestamp"] + timestamp_offset)[
                "temperature"
            ]
            .resample(freq)
            .mean()
        )

        for i in [load_profile_resampled, load_shape]:
            i["temperature"] = temp_resampled.values

    # create new dims for shape type
    load_profile_resampled["shape.type"] = "Load Profile"
    load_shape["shape.type"] = "Load Shape"

    load_shape_xr = xr.Dataset.from_dataframe(
        pd.concat([load_profile_resampled, load_shape]).set_index(
            ["shape.type"] + meta_cols, append=True
        )
    )

    load_shape_xr = load_shape_xr.assign_coords(
        timestamp=(freq_dict[freq][1], timestamp)
    )

    return (label, load_shape_xr)


def load_shape_from_multiple_load_profiles(
    load_profiles: Tuple[str, pd.DataFrame],
    timestamp_offset: Optional[pd.Timedelta] = None,
    meta_filt: List[str] = ["puma", "building_type"],
    values_filt: List[str] = ["out.electricity"],
    freq: str = "H",
    concat: bool = True,
    attach_temp: bool = True,
    agg_cols: Optional[List[Dict[str, Union[list, str]]]] = None,
) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
    """
    Wrapper function to handle multiple load profiles returned from read_profiles_from_csvs
    Requierd params:
    ...
    """
    xarrays = []
    for i, x in load_profiles.items():
        xarrays.append(
            load_shape_from_load_profile(
                i,
                x,
                timestamp_offset,
                meta_filt,
                values_filt,
                freq,
                attach_temp,
                agg_cols,
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
        attach_temp: bool = True,
        agg_cols: Optional[List[Dict[str, Union[list, str]]]] = None,
        dir: Optional[str] = None,
    ):
        # initialize params
        # don't initialize load_profiles as attr to limit memory impacts
        self.timestamp_offset = timestamp_offset
        self.meta_filt = meta_filt
        self.values_filt = values_filt
        self.freq = freq
        self.attach_temp = attach_temp
        self.agg_cols = agg_cols

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
            attach_temp=self.attach_temp,
            agg_cols=self.agg_cols,
        )
        return load_shapes

    def to_netcdf(self, path: str) -> None:
        self.load_shapes.to_netcdf(path)
