import pytest
import pandas as pd
from pydantic import ValidationError
from enduse.oedi_tools import LoadProfilePathValidator, _pull_nrel_load_profiles


class TestOEDITools:

    # TODO rewrite tests to handle valid/invalid url
    # def test__pull_nrel_load_profiles(self):
    #     assert isinstance(
    #         _pull_nrel_load_profiles(
    #             "resstock", "tmy3", "multi-family_with_2_-_4_units", "WA", "g53011606"
    #         ),
    #         pd.DataFrame,
    #     )

    # def test_fail__pull_nrel_load_profiles(self):
    #     with pytest.warns(expected_warning=UserWarning):
    #         _pull_nrel_load_profiles(
    #             "abc", "tmy3", "multi-family_with_2_-_4_units", "WA", "g53011606"
    #         )

    def test_segment_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "abc",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            LoadProfilePathValidator(**vars)

    def test_weather_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "abc",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            LoadProfilePathValidator(**vars)

    def test_com_bldg_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "comstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            LoadProfilePathValidator(**vars)

    def test_res_bldg_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "warehouse",
                "state": "WA",
                "puma_code": "g53011606",
            }

            LoadProfilePathValidator(**vars)

    def test_state_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "abc",
                "puma_code": "g53011606",
            }

            LoadProfilePathValidator(**vars)

    def test_puma_code_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "53011606",
            }

            LoadProfilePathValidator(**vars)

