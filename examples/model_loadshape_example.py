from calendar import calendar
from datetime import timedelta
from matplotlib.pyplot import axis
import pandas as pd
from enduse.oedi_tools import LoadProfiles
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np

vars = {
    "segment": "resstock",
    "weather_type": "amy2018",
    "state": "WA",
    "puma_code": "g53011606",
    "bldg_types": ["single-family_detached"],
}

nrel_single_family = LoadProfiles(**vars)

test_col = "out.electricity.total.energy_consumption"
keep_cols = ["timestamp", "units_represented", test_col]

enduse_loadshape = list(nrel_single_family.load_profiles.values())[0][keep_cols]
enduse_loadshape["datetime"] = pd.to_datetime(
    enduse_loadshape["timestamp"], format="%Y-%m-%d %H:%M:%S"
)
enduse_loadshape["datetime"] = enduse_loadshape["datetime"] - timedelta(
    hours=3, minutes=15
)

enduse_loadshape = enduse_loadshape.set_index("datetime")
enduse_loadshape = enduse_loadshape.drop("timestamp", 1)

enduse_loadshape_norm = enduse_loadshape.filter(like="out.", axis=1).div(
    enduse_loadshape["units_represented"], axis=0
)

enduse_loadshape_norm = enduse_loadshape_norm.rename(columns={test_col: "KWH"})

enduse_loadshape_norm = enduse_loadshape_norm.resample("H").sum()

enduse_loadshape_norm.reset_index(inplace=True)

enduse_loadshape_norm["HOUR"] = enduse_loadshape_norm["datetime"].dt.hour
enduse_loadshape_norm["WKDAY"] = enduse_loadshape_norm["datetime"].dt.dayofweek

cal = calendar()
holidays = cal.holidays(
    start=enduse_loadshape_norm["datetime"].min(),
    end=enduse_loadshape_norm["datetime"].max(),
)

enduse_loadshape_norm["HOLIDAY"] = enduse_loadshape_norm["datetime"].isin(holidays)

temps = pd.read_csv("I:/FINANCE/FPU/Matt/2018amy.csv", parse_dates=["datetime"])

temps["datetime"] = temps["datetime"].dt.tz_localize(None)
enduse_loadshape_norm = enduse_loadshape_norm.merge(temps, on="datetime", how="left")

index = np.linspace(-3, 8757, num=8760)
cosine = np.cos(2 * np.pi * index / 8760)

enduse_loadshape_norm["COS"] = cosine


enduse_loadshape_norm["COS"].plot()
enduse_loadshape_norm["Temperature"].plot()
enduse_loadshape_norm["Temperature"] = enduse_loadshape_norm[
    "Temperature"
].interpolate()
enduse_loadshape_norm.isnull().values.any()


y = enduse_loadshape_norm["KWH"]
y = y / y.max()
X = enduse_loadshape_norm.drop("datetime", 1)
X = X.drop("KWH", axis="columns")
X["WKDAY"] = pd.Categorical(X["WKDAY"])
X["HOLIDAY"] = pd.Categorical(X["HOLIDAY"])

# all_splits = list(ts_cv.split(X, y))
# train_0, test_0 = all_splits[3]
# X.iloc[test_0]

# evaluate sklearn histogram gradient boosting algorithm for classification
from numpy import mean
from numpy import std
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import TimeSeriesSplit

# ts_cv = TimeSeriesSplit(n_splits=6, gap=0, max_train_size=1000, test_size=1000)

model = HistGradientBoostingRegressor(
    categorical_features=[
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)


X = pd.get_dummies(X, prefix=["WKDAY", "HOLIDAY"])

# cv_results = cross_validate(
#     model,
#     X,
#     y,
#     cv=ts_cv,
#     scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"]
#     # error_score='raise'
# )
# mae = -cv_results["test_neg_mean_absolute_error"]
# rmse = -cv_results["test_neg_root_mean_squared_error"]
# print(
#     f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
#     f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
# )

test_fit = HistGradientBoostingRegressor().fit(X, y)


def mae(og, pred):
    return mean(abs(og - pred))


def mse(og, pred):
    return mean((og - pred) ** 2)


def rmse(og, pred):
    return mse(og, pred) ** 0.5


import matplotlib.pyplot as plt


def fit_results(fit, X, y):
    print(f"MAE = {mae(y,fit.predict(X))}")
    print(f"RMSE = {rmse(y,fit.predict(X))}")
    print(f"R^2 = {fit.score(X,y)}")

    pred = fit.predict(X)
    pred = pd.Series(pred)
    plt.plot(y)
    plt.plot(pred)
    plt.show()


fit_results(test_fit, X, y)
