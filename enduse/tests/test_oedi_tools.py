import pytest
import pandas as pd
from pydantic import ValidationError
from enduse.oedi_tools import PullLoadShape, pull_nrel_load_shapes


class TestOEDITools:
    def test_pull_nrel_load_shapes(self):
        assert isinstance(
            pull_nrel_load_shapes(
                "resstock", "tmy3", "multi-family_with_2_-_4_units", "WA", "g53011606"
            ),
            pd.DataFrame,
        )

    def test_fail_pull_nrel_load_shapes(self):
        with pytest.warns(expected_warning=UserWarning):
            pull_nrel_load_shapes(
                "abc", "tmy3", "multi-family_with_2_-_4_units", "WA", "g53011606"
            )

    def test_segment_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "abc",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            PullLoadShape(**vars)

    def test_weather_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "abc",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            PullLoadShape(**vars)

    def test_com_bldg_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "comstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "g53011606",
            }

            PullLoadShape(**vars)

    def test_res_bldg_type_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "warehouse",
                "state": "WA",
                "puma_code": "g53011606",
            }

            PullLoadShape(**vars)

    def test_state_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "abc",
                "puma_code": "g53011606",
            }

            PullLoadShape(**vars)

    def test_puma_code_check(self):
        with pytest.raises(ValidationError):
            vars = {
                "segment": "resstock",
                "weather_type": "tmy3",
                "bldg_type": "mobile_home",
                "state": "WA",
                "puma_code": "53011606",
            }

            PullLoadShape(**vars)

