import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import openrouteservice
from tqdm import tqdm

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Own Classes
from Simulation_Routing import *
from Simulating import *
from Simulation_Routing_Matrix import *
from Simulation_Routing_Matrix_copy import *
from Simulation_Routing_Matrix_Batch import *


def simulation_run(
    ip,
    num_sim=3,
    max_num_resp=100,
    open_hr=14.0,
    prop_cfr=0.1,
    aed_filter=["Yes"],
    all_open=False
):
    """
    First responder simulations, saving each set of runs (with given parameters) to a csv
    """
    simulations = Simulation(ip, all_open)
    routing_matrix = RoutingSimulationMatrixBatch(ip, all_open)

    # Adjust decline rate based on time
    if 7.0 <= open_hr < 17.0:
        decline_rate = 0.76
    elif 17.0 <= open_hr < 23.0:
        decline_rate = 0.64
    else:
        decline_rate = 0.88

    time_of_day = "day" if 8.0 <= open_hr <= 18.0 else "night"

    print(
        f"Input values:\n"
        f"  opening_hour: {open_hr}\n"
        f"  decline_rate: {decline_rate}\n"
        f"  max_number_responder: {max_num_resp}\n"
        f"  filter_values: {aed_filter}\n"
        f"  filter_values: {all_open}\n"
        f"  time_of_day: {time_of_day}\n"
        f"  proportion: {prop_cfr}"
    )

    df_final = pd.DataFrame()
    x = 1

    while x <= num_sim:
        try:
            df = simulations.simulation_run(
                decline_rate,
                max_num_resp,
                open_hr,
                aed_filter,
                time_of_day,
                prop_cfr
            )

            df["filter_values"] = (
                ",".join(aed_filter) if isinstance(aed_filter, list) else aed_filter
            )
            df["opening_hour"] = open_hr
            df["decline_rate"] = decline_rate
            df["max_number_responder"] = max_num_resp
            df["proportion_of_CFR"] = prop_cfr

            suffix = f"_run{x}"
            df_renamed = df.rename(
                columns={
                    col: f"{col}{suffix}" for col in df.columns
                    if col != "patient_loc"
                }
            )

            df_final.append(df_renamed)

        except Exception as e:
            print(f"[Warning] Simulation {x} failed: {e}. Retrying...")

        x += 1

    df_final = pd.concat(df_final, ignore_index=True)

    aed_247_str = "-open247" if all_open is True else "-opennormal"
    aed_str = "-all" if aed_filter is None else "-".join(aed_filter)
    filename = (
        f"sim_results_"
        f"openhr{open_hr}_"
        f"cfr{prop_cfr}_"
        f"maxresp{max_num_resp}_"
        f"numsim{num_sim}_"
        f"aed{aed_str}_"
        f"{aed_247_str}.csv"
    )

    df_final.to_csv(filename, index=False)
    print(f"Results saved to {filename}")



def main():
    ip = "35.159.16.25"

    num_sims = 3
    props_cfrs = [
        0.0025, 0.005, 0.0075, 0.01,
        0.0125, 0.015, 0.0175, 0.02,
        0.0225, 0.025, 0.0275, 0.03
    ]
    opening_hours = [2.0, 8.0, 14.0, 20.0]
    max_responders = [100, 150]
    aeds_all_open = [True, False]

    for prop_cfr in props_cfrs:
        for open_hr in opening_hours:
            for max_resp in max_responders:
                for open_setting in aeds_all_open:
                    simulation_run(
                        ip=ip,
                        num_sim=num_sims,
                        max_num_resp=max_resp,
                        open_hr=open_hr,
                        prop_cfr=prop_cfr,
                        all_open=open_setting
                    )


if __name__ == "__main__":
    main()

