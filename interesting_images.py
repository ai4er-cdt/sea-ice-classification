import numpy as np
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(description="Interesting Image Analysis")
    parser.add_argument("--tile_info_filename", type=str, help="Filename of tile_info csv to process")
    args = parser.parse_args()

    df = pd.read_csv(args.tile_info_filename)
    numeric_columns = [col for col in df.columns if col[0].isnumeric()]
    df["low"] = 0
    df["mid"] = 0
    df["high"] = 0
    for col in numeric_columns:
        if float(col) < 2:  # this is a low ice concentration column
            df["low"] += df[col]
        elif float(col) < 78:  # this is a mid ice concentration column
            df["mid"] += df[col]
        else:
            df["high"] += df[col]
    df["low_mid"] = np.logical_and(df["low"] > 0, df["mid"] > 0)
    df["low_mid_min"] = df[["low", "mid"]].min(axis=1)
    df["mid_high"] = np.logical_and(df["mid"] > 0, df["high"] > 0)
    df["mid_high_min"] = df[["mid", "high"]].min(axis=1)
    df["three"] = np.logical_and(df["low"] > 0, df["mid"] > 0, df["high"] > 0)
    df["three_min"] = df[["low", "mid", "high"]].min(axis=1)

    # generate output with all three classes
    df_three = df[df["three"]]
    df_three = df_three.sort_values(by="three_min", axis=0, ascending=False)
    df_three.to_csv(f"{args.tile_info_filename[:-4]}_three.csv")

    # generate output with low/mid classes
    df_low_mid = df[df["low_mid"]]
    df_low_mid = df_low_mid.sort_values(by="low_mid_min", axis=0, ascending=False)
    df_low_mid.to_csv(f"{args.tile_info_filename[:-4]}_low_mid.csv")

    # generate output with mid/high classes
    df_mid_high = df[df["mid_high"]]
    df_mid_high = df_mid_high.sort_values(by="mid_high_min", axis=0, ascending=False)
    df_mid_high.to_csv(f"{args.tile_info_filename[:-4]}_mid_high.csv")
