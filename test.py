"""
AI4ER GTC - Sea Ice Classification
Script for feeding test data into unet or 
resnet34 model and saving the model output to wandb
"""
import pytorch_lightning as pl
import wandb
import pandas as pd
import segmentation_models_pytorch as smp
from argparse import ArgumentParser
from constants import new_classes
from torch.utils.data import DataLoader
from torch import nn
from util import SeaIceDataset, Visualise
from model import UNet, Segmentation
from pathlib import Path

if __name__ == "__main__":

    parser = ArgumentParser(description="Sea Ice Segmentation Test")
    parser.add_argument("--username", type=str, help="wandb username")
    parser.add_argument("--name", type=str, help="Name of wandb run")
    parser.add_argument("--checkpoint", type=str, help="Name of checkpoint file")
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    parser.add_argument("--test_mode", default="All", type=str, choices=["Single", "Interesting", "All"],
                        help="Test on single, interesting, or all images")
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--n_workers", default=1, type=int, help="Number of workers in dataloader")
    parser.add_argument("--val_tile_info_base", default="tile_info_13032023T164009",
                        type=str, help="Tile info csv to load images for visualisation")
    parser.add_argument("--test_tile_info_base", default="tile_info_13032023T230145",
                        type=str, help="Tile info csv to load images for visualisation")
    parser.add_argument("--n_to_visualise", default=3, type=int, help="How many tiles per category to visualise")
    args = parser.parse_args()

    # wandb logging
    wandb.init(id=args.name, project="sea-ice-classification", resume="must")
    api = wandb.Api()
    run = api.run(f"{args.username}/sea-ice-classification/{args.name}")
    wandb_logger = pl.loggers.WandbLogger(project="sea-ice-classification")

    # load (most) command line args from original training run config file
    del run.config["name"]  # keep name from this run's flags
    del run.config["accelerator"]  # keep accelerator choice from this run's flags
    del run.config["devices"]  # keep devices choice from this run's flags
    del run.config["n_workers"]  # keep n_workers choice from this run's flags
    vars(args).update(run.config)

    # standard input dirs
    val_tile_folder = f"{open('tile.config').read().strip()}"
    tile_folder = f"{val_tile_folder}/test"
    sar_folder = f"{tile_folder}/sar"
    chart_folder = f"{tile_folder}/chart"

    # get file lists
    if args.test_mode == "Single":  # load single test file
        test_files = ["WS_20221216_00001_[12160,128]_256x256.tiff"] * args.batch_size * 2  # TODO replace
    elif args.test_mode == "Interesting":  # load a few interesting test pairs
        df = pd.read_csv("interesting_test_images.csv")[:5]  # TODO
        files = []
        for i, row in df.iterrows():
            files.append(
                f"{row['region']}_{row['basename']}_{row['file_n']:05}_[{row['col']},{row['row']}]_{row['size']}x{row['size']}.tiff")
        test_files = files
    else:  # load full sets of test files from pre-determined lists
        with open(Path(f"{tile_folder}/test_files.txt"), "r") as f:
            test_files = f.read().splitlines()
    print(f"Length of test file list {len(test_files)}.")

    # get val visualisation file lists
    val_dfs = {
        "low": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_low.csv", index_col=0)[:args.n_to_visualise],
        "mid": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_mid.csv", index_col=0)[:args.n_to_visualise],
        "high": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_high.csv", index_col=0)[:args.n_to_visualise],
        "low_mid": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_low_mid.csv", index_col=0)[:args.n_to_visualise],
        "mid_high": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_mid_high.csv", index_col=0)[:args.n_to_visualise],
        "low_high": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_low_high.csv", index_col=0)[:args.n_to_visualise],
        "three": pd.read_csv(f"{val_tile_folder}/{args.val_tile_info_base}_three.csv", index_col=0)[:args.n_to_visualise]
    }
    val_vis_files = []
    for df in val_dfs.values():
        if len(df) > 0:
            val_vis_files.extend(df["filename"].to_list())
    print(f"Length of val vis file list {len(val_vis_files)}.")

    # get test visualisation file lists
    test_dfs = {
        "low": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_low.csv", index_col=0)[:args.n_to_visualise],
        "mid": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_mid.csv", index_col=0)[:args.n_to_visualise],
        "high": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_high.csv", index_col=0)[:args.n_to_visualise],
        "low_mid": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_low_mid.csv", index_col=0)[:args.n_to_visualise],
        "mid_high": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_mid_high.csv", index_col=0)[:args.n_to_visualise],
        "low_high": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_low_high.csv", index_col=0)[:args.n_to_visualise],
        "three": pd.read_csv(f"{tile_folder}/{args.test_tile_info_base}_three.csv", index_col=0)[:args.n_to_visualise]
    }
    test_vis_files = []
    for df in test_dfs.values():
        if len(df) > 0:
            test_vis_files.extend(df["filename"].to_list())
    print(f"Length of test vis file list {len(test_vis_files)}.")


    # init
    pl.seed_everything(args.seed)
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)

    # load test data
    test_sar_files = [f"SAR_{f}" for f in test_files]
    test_chart_files = [f"CHART_{f}" for f in test_files]
    test_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=test_sar_files,
                                 chart_path=chart_folder, chart_files=test_chart_files,
                                 class_categories=class_categories)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

    # load val vis data
    val_vis_sar_files = [f"SAR_{f}" for f in val_vis_files]
    val_vis_chart_files = [f"CHART_{f}" for f in val_vis_files]
    val_vis_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=val_vis_sar_files,
                                    chart_path=chart_folder, chart_files=val_vis_chart_files,
                                    class_categories=class_categories)
    val_vis_dataloader = DataLoader(val_vis_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

    # load test vis data
    test_vis_sar_files = [f"SAR_{f}" for f in test_vis_files]
    test_vis_chart_files = [f"CHART_{f}" for f in test_vis_files]
    test_vis_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=test_vis_sar_files,
                                     chart_path=chart_folder, chart_files=test_vis_chart_files,
                                     class_categories=class_categories)
    test_vis_dataloader = DataLoader(test_vis_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

    # configure model
    if args.model == "unet":
        model = UNet(kernel=3, n_channels=3, n_filters=args.n_filters, n_classes=n_classes)
    else:  # assume unet encoder from segmentation_models_pytorch (see smp documentation for valid strings)
        decoder_channels = [2 ** (i + 4) for i in range(args.encoder_depth)][::-1]  # eg [64,32,16] for encoder_depth=3
        model = smp.Unet(args.model, encoder_weights="imagenet",
                         encoder_depth=args.encoder_depth,
                         decoder_channels=decoder_channels,
                         in_channels=3, classes=n_classes)

    # configure loss
    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "dice":
        criterion = smp.losses.DiceLoss(mode="multiclass")
    elif args.criterion == "focal":
        criterion = smp.losses.FocalLoss(mode="multiclass")
    else:
        raise ValueError(f"Invalid loss function: {args.criterion}.")

    # load model from best checkpoint
    checkpoint_path = Path(f"./sea-ice-classification/{args.name}/checkpoints/{args.checkpoint}")
    segmenter = Segmentation.load_from_checkpoint(checkpoint_path, model=model, criterion=criterion)

    # test
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(Visualise(val_vis_dataloader, len(val_vis_files), args.classification_type))
    trainer.callbacks.append(Visualise(test_vis_dataloader, len(test_vis_files), args.classification_type))

    # train model
    print(f"Testing {len(test_dataset)} examples / {len(test_dataloader)} batches (batch size {args.batch_size}).")
    print(f"All arguments: {args}")
    trainer.test(segmenter, test_dataloader)
