import pytorch_lightning as pl
import wandb
from constants import new_classes
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset, Visualise
from model import Segmentation, UNet
from torchmetrics import JaccardIndex
from pathlib import Path
import segmentation_models_pytorch as smp

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Sea Ice Segmentation Train")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="unet", type=str,
                        help="Either 'unet' or smp decoder (e.g.'densenet201','vgg19','resnet34','resnext50_32x4d'), "
                             "see https://segmentation-modelspytorch.readthedocs.io/en/latest", required=False)
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--overfit", default=False, type=eval, help="Whether or not to overfit on a single image")
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
    args = parser.parse_args()

    # standard input dirs
    tile_folder = open("tile.config").read().strip()
    chart_folder = f"{tile_folder}/chart"
    sar_folder = f"{tile_folder}/sar"


    # get file lists
    if args.overfit:  # load single train/val file and overfit
        train_files = ["AP_20181202_00040_[9216,512]_256x256.tiff"] * args.batch_size * 100
        val_files = ["AP_20181202_00040_[9216,512]_256x256.tiff"] * args.batch_size
    else:  # load full sets of train/val files from pre-determined lists
        with open(Path(f"{tile_folder}/train_files.txt"), "r") as f:
            train_files = f.read().splitlines()
        with open(Path(f"{tile_folder}/val_files.txt"), "r") as f:
            val_files = f.read().splitlines()

    # init
    pl.seed_everything(args.seed)
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    decoder_channels = [2 ** (i + 4) for i in range(args.encoder_depth)][::-1]  # e.g. [64,32,16] for encoder_depth = 3

    # load training data
    train_sar_files = [f"SAR_{f}" for f in train_files]
    train_chart_files = [f"CHART_{f}" for f in train_files]
    train_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=train_sar_files,
                                  chart_path=chart_folder, chart_files=train_chart_files,
                                  class_categories=class_categories, sar_band3=args.sar_band3)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # load validation data
    val_sar_files = [f"SAR_{f}" for f in val_files]
    val_chart_files = [f"CHART_{f}" for f in val_files]
    val_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=val_sar_files,
                                chart_path=chart_folder, chart_files=val_chart_files,
                                class_categories=class_categories, sar_band3=args.sar_band3)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # configure model
    if args.model == "unet":
        model = UNet(kernel=3, n_channels=3, n_filters=args.n_filters, n_classes=n_classes)
    else:  # assume unet encoder from segmentation_models_pytorch (see smp documentation for valid strings)
        model = smp.Unet(args.model, encoder_weights='imagenet',
                         encoder_depth=args.encoder_depth,
                         decoder_channels=decoder_channels,
                         in_channels=3, classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    metric = JaccardIndex(task="multiclass", num_classes=n_classes)
    segmenter = Segmentation(model, criterion, args.learning_rate, metric)

    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="sea-ice-classification")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(args)

    # set up trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="val_loss"))
    trainer.callbacks.append(Visualise(val_dataloader))

    # train model
    trainer.fit(segmenter, train_dataloader, val_dataloader)
