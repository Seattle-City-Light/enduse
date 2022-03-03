import numpy as np
import pandas as pd

from typing import List, Type

from enduse.stockobjects_tabular import (
    CalcConfig,
    StockObject,
    CommonStockObject,
    BuildingObject,
)


class StockTurnoverModel:
    def __init__(self, stock_objects: Type[StockObject], calc_config: Type[CalcConfig]):
        self.buildings: pd.DataFrame = None
        self.saturation: pd.DataFrame = None
        self.fuel_share: pd.DataFrame = None
        self.efficiency_share: pd.DataFrame = None
        self.equipment_measures: pd.DataFrame = None
        self.ramp_efficiency: pd.DataFrame = None

        self.output = self.run_model(stock_objects, calc_config)

    def run_model(
        self, stock_objects: Type[StockObject], calc_config: Type[CalcConfig]
    ) -> pd.DataFrame:

        # set dataframe class attributes
        self.parse_stock_object(stock_objects)

        # run model
        model_data = self.join_stock_objects(self.build_join_order(stock_objects))

        model_data = model_data.sort_values(by=list(calc_config.equipment_sort_index))

        model_data[calc_config.equipment_label] = (
            model_data[list(calc_config.equipment_calc_variables)]
            .cumprod(axis=1)
            .iloc[:, -1]
        )

        # will hold turnover calculations
        turnover_list = []

        # loop over groups to calculate stock turnover for each equipment type
        for idx, grp in model_data.groupby(list(calc_config.equipment_calc_index)):

            turnover = self.stock_turnover_calc(
                grp[calc_config.equipment_label].values,
                grp[calc_config.effective_useful_life_label].values,
                grp[calc_config.efficiency_level_label].values,
                grp[calc_config.ramp_efficiency_level_label].values,
                grp[calc_config.ramp_efficiency_probability_label].values,
            )

            turnover_list.append(turnover)

        # convert list of equipment turnovers to matrix and merge to dataframe
        model_data[calc_config.equipment_stock_label] = np.hstack(turnover_list)

        # calculate energy estimate
        model_data[calc_config.equipment_consumption_label] = (
            model_data[calc_config.equipment_stock_label]
            * model_data[calc_config.equipment_measure_consumption_label]
            / 1000
        )

        model_data = model_data.reset_index(drop=True)

        return model_data.drop([calc_config.equipment_label], axis=1)

    def parse_stock_object(self, stock_objects: Type[StockObject]):
        for i in stock_objects.__fields__:
            if i == "buildings":
                setattr(self, i, create_buildings_dataframe(getattr(stock_objects, i)))
            else:
                setattr(
                    self, i, create_common_files_dataframe(getattr(stock_objects, i))
                )

    def build_join_order(self, stock_objects: Type[StockObject]):
        """Instantiate join order attr"""
        join_order = []

        for i in stock_objects.__fields__:
            join_order.append(
                (
                    i,
                    getattr(getattr(stock_objects, i), "join_order"),
                    getattr(getattr(stock_objects, i), "join_on"),
                )
            )

        join_order = sorted(join_order, key=lambda x: x[1])

        return join_order

    def join_stock_objects(self, join_order):
        for n, (i, x) in enumerate(zip(join_order[:-1], join_order[1:])):
            if n == 0:
                model_data = pd.merge(
                    getattr(self, i[0]), getattr(self, x[0]), how="left", on=i[2]
                )
            else:
                model_data = model_data.merge(getattr(self, x[0]), how="left", on=x[2])

        return model_data

    def stock_turnover_calc(
        self,
        stock: np.ndarray,
        eul: np.ndarray,
        e_level: np.ndarray,
        se_level: np.ndarray,
        ramp_prob: np.ndarray,
    ) -> np.ndarray:

        """Custom function for stock turnover calculation from a groupby"""

        # get count of efficiency type
        e_levels = np.unique(e_level).shape[0]

        # these matrices will hold total stock and turnover
        s_mat = np.zeros([np.int(len(stock) / e_levels), e_levels])
        t_mat = np.zeros([np.int(len(stock) / e_levels), e_levels])
        s_mat_new = np.zeros(e_levels)

        # reshape input matrices for useful life and efficiency levels
        stock_mat = np.reshape(stock, np.flip(s_mat.shape)).T
        eul_mat = np.reshape(eul, np.flip(s_mat.shape)).T
        e_mat = np.reshape(e_level, np.flip(s_mat.shape)).T
        se_mat = np.reshape(se_level, np.flip(s_mat.shape)).T
        prob_mat = np.reshape(ramp_prob, np.flip(s_mat.shape)).T

        # set initial stock values
        s_mat[0, :] = np.rint(stock_mat[0, :])

        # loop over each forecast year [i]
        for i in range(s_mat.shape[0] - 1):
            # loop over each efficiency type [j]
            t_mat[i + 1, :] = np.rint(s_mat[i, :] / eul_mat[i, :])
            # check if there is any change in stock from new buildings or saturation changes
            s_mat_new = np.diff(np.rint(stock_mat), axis=0)[i]
            for j in range(s_mat.shape[1]):
                # if efficiency type [j] is less than standard then convert to standard efficiency type
                if se_mat[i + 1, j] > e_mat[i, j]:
                    offset = np.int(se_mat[i + 1, j] - e_mat[i, j])
                    # need to make sure that transition probably is factored in when accounting for new building stock
                    # this calculates the amount of below standard stock that will not turnover
                    s_mat[i + 1, j] = np.rint(
                        (s_mat[i, j] - t_mat[i + 1, j] * prob_mat[i + 1, j])
                        + s_mat_new[j] * (1 - prob_mat[i + 1, j])
                    )
                    # this calculates the amount of below standard stock that will turnover to higher efficiency
                    s_mat[i + 1, j + offset] = np.rint(
                        (
                            s_mat[i + 1, j + offset]
                            + t_mat[i + 1, j] * prob_mat[i + 1, j]
                            + prob_mat[i + 1, j] * s_mat_new[j]
                        )
                    )
                # if efficiency type [j] is above standard then replace with itself
                else:
                    s_mat[i + 1, j] = s_mat[i, j] + s_mat[i + 1, j] + s_mat_new[j]

        return np.ravel(s_mat, order="F")


def create_common_files_dataframe(
    stock_object: Type[CommonStockObject], df: pd.DataFrame = None
) -> pd.DataFrame:

    # added optional df parameter for testing
    if df is None:
        df = pd.read_csv(stock_object.file_path)

    # define columns used for interpolation
    df_idx = [
        stock_object.interpolation_label,
        stock_object.start_time,
        stock_object.end_time,
        stock_object.start_variable,
        stock_object.end_variable,
    ]

    interpolated_data = [
        interpolate_dataframe(i[0], i[1], i[2], i[3], i[4]) for i in df[df_idx].values
    ]

    df[stock_object.start_time] = pd.to_datetime(
        df[stock_object.start_time], format="%Y"
    )
    df[stock_object.end_time] = pd.to_datetime(df[stock_object.end_time], format="%Y")

    # create standard date ranges
    dates = [
        pd.date_range(r[0], r[1], freq="AS")
        for r in df[[stock_object.start_time, stock_object.end_time]].values
    ]

    # reshape data frame and load with range and probabilty interpolation
    lens = [len(x) for x in dates]

    # create reshaped dataframe
    df_long = pd.DataFrame({col: np.repeat(df[col].values, lens) for col in df.columns})
    df_long[stock_object.time_label] = pd.to_datetime(np.concatenate(dates)).year
    df_long[stock_object.label] = np.concatenate(interpolated_data)

    # # drop duplicate colummns
    df_long = df_long.drop(
        [
            stock_object.start_time,
            stock_object.end_time,
            stock_object.start_variable,
            stock_object.end_variable,
        ],
        axis=1,
    )

    return df_long


# buildings do not get interpolated
def create_buildings_dataframe(
    stock_object: Type[BuildingObject], df: pd.DataFrame = None
) -> pd.DataFrame:

    if df is None:
        df = pd.read_csv(stock_object.file_path)

    return df


def interpolate_dataframe(
    ramp: str, start_year: int, end_year: int, start_value: float, end_value: float
) -> np.array:

    # other custom funcation can be added here
    interpolation_functions = {"linear": np.linspace}

    interpolated_values = interpolation_functions[ramp](
        start_value, end_value, (end_year - start_year + 1)
    )

    return interpolated_values
