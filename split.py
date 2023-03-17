"""
AI4ER GTC - Sea Ice Classification
Functions to create the training, validation 
& testing datasets
"""
import os
import random
import pandas as pd
from random import shuffle
from argparse import ArgumentParser
from pathlib import Path


def construct_train_val(tile_info_csv):
    """
    Construct train/val set splits with filenames into .txt based on tile_info CSV
    """
    tile_directory = Path(open("tile.config").read().strip())
    train, val = [], []
    table = pd.read_csv(str(tile_directory / tile_info_csv))
    for i, row in table.iterrows():
        if row["basename"] in (20171106, 20190313, 20200117):  # select 3 specific WS images for validation
            val.append(f"{row['region']}_{row['basename']}_{row['file_n']:05}_[{row['col']},{row['row']}]_{row['size']}x{row['size']}.tiff")
        else:  # use all other images in training
            train.append(f"{row['region']}_{row['basename']}_{row['file_n']:05}_[{row['col']},{row['row']}]_{row['size']}x{row['size']}.tiff")
    random.seed(0)
    shuffle(train)  # ensure our train/val split is reproducibly random
    shuffle(val)  # ensure our train/val split is reproducibly random
    with open(tile_directory / "train_files.txt", "w") as f:
        f.write("\n".join(train))
    with open(tile_directory / "val_files.txt", "w") as f:
        f.write("\n".join(val))


def construct_test(tile_info_csv):
    """
    Construct test set with filenames into .txt based on tile_info CSV
    """
    tile_directory = Path(open("tile.config").read().strip()) / "test"
    test = []
    table = pd.read_csv(str(tile_directory / tile_info_csv))
    for i, row in table.iterrows():
        test.append(f"{row['region']}_{row['basename']}_{row['file_n']:05}_[{row['col']},{row['row']}]_{row['size']}x{row['size']}.tiff")
    random.seed(0)
    shuffle(test)  # ensure our test set is reproducibly random
    with open(tile_directory / "test_files.txt", "w") as f:
        f.write("\n".join(test))


if __name__ == "__main__":

    parser = ArgumentParser(description="Sea Ice Train/Val/Test Split")
    parser.add_argument("--mode", default="train/val", type=str, choices=["train/val", "test"],
                        help="Whether to split train/val images or test images")
    parser.add_argument("--tile_info_csv", type=str, help="Which tile_info_csv to pull from")
    args = parser.parse_args()

    if args.mode == "train/val":
        construct_train_val(args.tile_info_csv)
    else:
        construct_test(args.tile_info_csv)
