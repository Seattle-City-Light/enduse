import numpy as np
import pandas as pd

from typing import List

from matplotlib import pyplot as plt

from enduse.stockobjects import Equipment, RampEfficiency, EndUse, Building
from enduse.stockturnover import BuildingModel
from enduse.oedi_tools import PullLoadProfiles

oedi_sf_scl_parms = {
    "segment": "resstock",
    "weather_type": "tmy3",
    "bldg_type": "single-family_attached",
    "state": "WA",
    "puma_code": "g53011606",
}

oedi_sf_scl = PullLoadProfiles(**oedi_sf_scl_parms)
oedi_sf_scl_df = oedi_sf_scl.get_load_shapes()

agg_cols = [
    "out.electricity.cooling.energy_consumption",
    "out.electricity.heating.energy_consumption",
    "out.electricity.heating_supplement.energy_consumption",
]

oedi_sf_scl_df_norm = (
    oedi_sf_scl_df.set_index(
        oedi_sf_scl_df["timestamp"] - pd.Timedelta(hours=3, minutes=15)
    )
    .filter(agg_cols)
    .resample("H")
    .mean()
)

oedi_sf_scl_df_norm["resistence_heating"] = oedi_sf_scl_df_norm[agg_cols[-2:]].sum(
    axis=1
)

oedi_sf_scl_df_norm["heat_pump"] = oedi_sf_scl_df_norm[agg_cols].sum(axis=1)

oedi_sf_scl_df_norm = oedi_sf_scl_df_norm.transform(lambda x: x / x.sum())

equipment = []

equipment.append(
    {
        "equipment_label": "Standard Electric Furnace HSPF = 1",
        "efficiency_level": 1,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(1, 1, 20).tolist(),
        "consumption": np.linspace(13312, 13312, 20).tolist(),
        "useful_life": np.linspace(15, 15, 20).tolist(),
    }
)

equipment.append(
    {
        "equipment_label": "Install Ductless Heat Pump in House with Existing FAF - HZ1",
        "efficiency_level": 2,
        "start_year": 2022,
        "end_year": 2041,
        "efficiency_share": np.linspace(0, 0, 20).tolist(),
        "consumption": np.linspace(8500, 8500, 20).tolist(),
        "useful_life": np.linspace(18, 18, 20).tolist(),
    }
)

equipment_parsed = [Equipment(**i) for i in equipment]

ramp = {
    "ramp_label": "Forced Air Furnance to Heat Pump Upgrade",
    "ramp_equipment": [equipment_parsed[1]],
}

ramp_parsed = RampEfficiency(**ramp)

end_use = {
    "end_use_label": "Heat Central",
    "equipment": equipment_parsed,
    "ramp_efficiency": ramp_parsed,
    "saturation": np.linspace(0.50, 0.50, 20).tolist(),
    "fuel_share": np.linspace(0.05, 0.05, 20).tolist(),
}

end_use_parsed = [EndUse(**end_use)]

building = {
    "building_label": "Single Family",
    "end_uses": end_use_parsed,
    "building_stock": np.linspace(200000, 200000, 20).tolist(),
}

building_parsed = Building(**building)

stock_turnover = BuildingModel(building_parsed)

x = stock_turnover.model["consumption"].values[0]
y = oedi_sf_scl_df_norm[["resistence_heating", "heat_pump"]].values
z = stock_turnover.model["equipment_stock"].values[0]

cons_shaped = np.einsum("ij,ki->ijk", x, y)
cons_shaped_df = pd.DataFrame(
    cons_shaped.reshape(
        (cons_shaped.shape[0], cons_shaped.shape[1] * cons_shaped.shape[2])
    ).T
)

reindex = pd.date_range(start="2022-01-01 00", end="2041-12-31 23", freq="H")

cons_shaped_df = cons_shaped_df.set_index(
    reindex[~((reindex.month == 2) & (reindex.day == 29))]
)

cons_shaped_df = cons_shaped_df.rename(columns={0: "Electric Furnance", 1: "Heat Pump"})

fig, axs = plt.subplots(ncols=3, figsize=(14, 4), sharey=True)

for ax, year in zip(axs, ["2022", "2030", "2040"]):
    ax.set_title(f"Aggregate Load Shape: {year}", size=10, color="grey")
    ax.plot(
        cons_shaped_df.loc[year].sum(axis=1) / 1000,
        color="#FFBE0B",
        alpha=0.75,
        label="Electric Resistence Space Heat",
    )
    ax.plot(
        cons_shaped_df.loc[year]["Heat Pump"] / 1000,
        color="#3A86FF",
        alpha=0.75,
        label="Heat Pump",
    )

    # axis formatting
    ax.tick_params(color="grey", labelcolor="grey")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_edgecolor("lightgrey")
    ax.spines["bottom"].set_edgecolor("lightgrey")
    ax.spines["left"].set_edgecolor("lightgrey")
    ax.spines["right"].set_edgecolor("lightgrey")

    ax.set_xlabel("Hour of Year (8760)", size=10, color="grey")

# formatting
# axs[2].legend(frameon=False)
axs[0].set_ylabel("Hourly Load (MWh)", size=10, color="grey")

legend = axs[2].legend(loc="upper left", frameon=False)

for text in legend.get_texts():
    text.set_color("grey")
    text.set_size(10)

fig.tight_layout()

fig.savefig("I:/FINANCE/FPU/LOAD/2022/Requests/NREL/load_shape_graphic.png", dpi=300)
