"""
AI4ER GTC - Sea Ice Classification
Script for feeding training and validation data into 
unet or resnet34 model and saving the model output to wandb
"""
import pandas as pd
import pytorch_lightning as pl
import wandb
from constants import new_classes
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset, Visualise
from model import Segmentation, UNet
from pathlib import Path
import segmentation_models_pytorch as smp


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Sea Ice Segmentation Train")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="unet", type=str,
                        help="Either 'unet' or smp decoder 'resnet34'"
                             "see https://segmentation-modelspytorch.readthedocs.io/en/latest", required=False)
    parser.add_argument("--criterion", default="ce", type=str, choices=["ce", "dice", "focal"],
                        help="Loss to train with", required=False)
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--user_overfit", default="False", type=str, choices=["True", "Semi", "False"],
                        help="Whether or not to overfit on a single image")
    parser.add_argument("--user_overfit_batches", default=5, type=int,
                        help="How many batches to run per epoch when overfitting")
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--n_workers", default=1, type=int, help="Number of workers in dataloader")
    parser.add_argument("--n_filters", default=16, type=int,
                        help="Number of convolutional filters in hidden layer if model==unet")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    parser.add_argument("--precision", default=32, type=int, help="Precision for training. Options are 32 or 16")
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="How often to log during training")
    parser.add_argument("--encoder_depth", default=5, type=int,
                        help="Number of decoder stages for smp models (increases number of features)")
    parser.add_argument("--max_epochs", default=100, type=int, help="Number of epochs to fine-tune")
    parser.add_argument("--num_sanity_val_steps", default=2, type=int, help="Number of batches to sanity check before training")
    parser.add_argument("--limit_train_batches", default=1.0, type=float, help="Proportion of training dataset to use")
    parser.add_argument("--limit_val_batches", default=1.0, type=float, help="Proportion of validation dataset to use")
    parser.add_argument("--tile_info_base", default="tile_info_13032023T164009",
                        type=str, help="Tile info csv to load images for visualisation")
    parser.add_argument("--n_to_visualise", default=3, type=int, help="How many tiles per category to visualise")
    args = parser.parse_args()

    # standard input dirs
    tile_folder = open("tile.config").read().strip()
    chart_folder = f"{tile_folder}/chart"
    sar_folder = f"{tile_folder}/sar"

    # get file lists
    if args.user_overfit == "True":  # load single train/val file and overfit
        train_files = ["WS_20180104_02387_[3840,4352]_256x256.tiff"] * args.batch_size * args.user_overfit_batches
        val_files = ["WS_20180104_02387_[3840,4352]_256x256.tiff"] * args.batch_size * 2
    elif args.user_overfit == "Semi":  # load a few interesting train/val pairs
        df = pd.read_csv("interesting_images.csv")[:5]
        files = []
        for i, row in df.iterrows():
            files.append(f"{row['region']}_{row['basename']}_{row['file_n']:05}_[{row['col']},{row['row']}]_{row['size']}x{row['size']}.tiff")
        train_files = files * args.batch_size * (args.user_overfit_batches // 5)
        val_files = files
    else:  # load full sets of train/val files from pre-determined lists
        with open(Path(f"{tile_folder}/train_files.txt"), "r") as f:
            train_files = f.read().splitlines()
        with open(Path(f"{tile_folder}/val_files.txt"), "r") as f:
            val_files = f.read().splitlines()
    print(f"Length of train file list {len(train_files)}.")
    print(f"Length of val file list {len(val_files)}.")

    # get visualisation file lists
    dfs = {
        "low": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_low.csv", index_col=0)[:args.n_to_visualise],
        "mid": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_mid.csv", index_col=0)[:args.n_to_visualise],
        "high": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_high.csv", index_col=0)[:args.n_to_visualise],
        "low_mid": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_low_mid.csv", index_col=0)[:args.n_to_visualise],
        "mid_high": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_mid_high.csv", index_col=0)[:args.n_to_visualise],
        "low_high": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_low_high.csv", index_col=0)[:args.n_to_visualise],
        "three": pd.read_csv(f"{tile_folder}/{args.tile_info_base}_three.csv", index_col=0)[:args.n_to_visualise]
    }
    val_vis_files = []
    for df in dfs.values():
        if len(df) > 0:
            val_vis_files.extend(df["filename"].to_list())
    print(f"Length of validation vis file list {len(val_vis_files)}.")

    # init
    pl.seed_everything(args.seed)
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)

    # load training data
    train_sar_files = [f"SAR_{f}" for f in train_files]
    train_chart_files = [f"CHART_{f}" for f in train_files]
    train_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=train_sar_files,
                                  chart_path=chart_folder, chart_files=train_chart_files,
                                  class_categories=class_categories, sar_band3=args.sar_band3)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

    # load validation data
    val_sar_files = [f"SAR_{f}" for f in val_files]
    val_chart_files = [f"CHART_{f}" for f in val_files]
    val_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=val_sar_files,
                                chart_path=chart_folder, chart_files=val_chart_files,
                                class_categories=class_categories, sar_band3=args.sar_band3)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

    # load validation vis data
    val_vis_sar_files = [f"SAR_{f}" for f in val_vis_files]
    val_vis_chart_files = [f"CHART_{f}" for f in val_vis_files]
    val_vis_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=val_vis_sar_files,
                                    chart_path=chart_folder, chart_files=val_vis_chart_files,
                                    class_categories=class_categories, sar_band3=args.sar_band3)
    val_vis_dataloader = DataLoader(val_vis_dataset, batch_size=args.batch_size, num_workers=args.n_workers, persistent_workers=True)

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

    # configure PyTorch Lightning module
    segmenter = Segmentation(model, n_classes, criterion, args.learning_rate)

    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="sea-ice-classification")
    wandb_logger.experiment.config.update(args)

    # turn off gradient logging to enable gpu parallelisation (wandb cannot parallelise when tracking gradients)
    # wandb_logger.watch(model, log="all", log_freq=10)

    # set up trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="val_loss"))
    trainer.callbacks.append(Visualise(val_vis_dataloader, len(val_vis_files), args.classification_type))

    # train model
    print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {args.batch_size}).")
    print(f"Validating {len(val_dataset)} examples / {len(val_dataloader)} batches (batch size {args.batch_size}).")
    print(f"All arguments: {args}")
    trainer.fit(segmenter, train_dataloader, val_dataloader)
