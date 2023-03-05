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
    parser.add_argument("--model", default="unet", type=str, choices=["unet", "densenet"],
                        help="Name of model to train", required=True)
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--n_workers", default=1, type=int, help="Number of workers in dataloader")
    parser.add_argument("--n_filters", default=16, type=int,
                        help="Number of convolutional filters in hidden layer if model==unet")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=100, type=int, help="Number of epochs to fine-tune")
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    parser.add_argument("--precision", default=32, type=int, help="Precision for training. Options are 32 or 16")
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="How often to log during training")
    parser.add_argument("--overfit", default=False, type=eval, help="Whether or not to overfit on a single image")
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"],
                        help="Binary, ternary or multiclass classification")
    args = parser.parse_args()

    # standard input dirs
    tile_folder = open("tile.config").read().strip()
    sar_folder = f"{tile_folder}/sar"
    chart_folder = f"{tile_folder}/chart"

    # get file lists
    if args.overfit:  # load single train/val file and overfit
        train_files = val_files = ["AP_20181202_00040_[9216,512]_256x256.tiff"] * args.batch_size
        args.max_epochs = 1000
    else:  # load full sets of train/val files from pre-determined lists
        with open(Path(f"{tile_folder}/train_files.txt"), "r") as f:
            train_files = f.read().splitlines()
        with open(Path(f"{tile_folder}/val_files.txt"), "r") as f:
            val_files = f.read().splitlines()

    # init
    pl.seed_everything(args.seed)
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)

    # load training data
    train_sar_files = [f"SAR_{f}" for f in train_files]
    train_chart_files = [f"CHART_{f}" for f in train_files]
    train_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=train_sar_files,
                                  chart_path=chart_folder, chart_files=train_chart_files,
                                  transform=None, class_categories=class_categories)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.n_workers)

    # load validation data
    val_sar_files = [f"SAR_{f}" for f in val_files]
    val_chart_files = [f"CHART_{f}" for f in val_files]
    val_dataset = SeaIceDataset(sar_path=sar_folder, sar_files=val_sar_files,
                                chart_path=chart_folder, chart_files=val_chart_files,
                                transform=None, class_categories=class_categories)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=args.n_workers)

    # configure model
    if args.model == "unet":
        model = UNet(kernel=3, n_channels=3, n_filters=args.n_filters, n_classes=n_classes)
    elif args.model == "densenet":
        model = smp.Unet('densenet201', encoder_weights='imagenet', encoder_depth=1, decoder_channels=[16], in_channels=3, classes=n_classes)
    elif args.model == "vgg19":
        model = smp.Unet('vgg19', encoder_weights='imagenet', encoder_depth=1, decoder_channels=[16], in_channels=3, classes=n_classes)

    else:
        raise ValueError("Unsupported model type")
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
