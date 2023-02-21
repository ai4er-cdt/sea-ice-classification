import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset
from pathlib import Path
from model import Segmentation, UNet
from torchmetrics import JaccardIndex  # may be called Jaccard Index in newer versions of torchmetrics


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Sea Ice Segmentation")
    parser.add_argument("--name", default="seaice", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="unet", type=str, help="Name of model to train")
    parser.add_argument('--gpu_id', default=-1, type=int, help="GPU id to train on")
    parser.add_argument('--n_filters', default=16, type=float, help="Number of convolutional filters in hidden layer")
    parser.add_argument('--learning_rate', default=0.05, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--max_epochs', default=15, type=int, help="Number of epochs to fine-tune")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    args = parser.parse_args()

    if (args.gpu_id == -1):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    base_folder = open("data_path.config").read().strip()
    chart_folder = Path(f"../Tiled_images/binary_chart")
    sar_folder = Path(f"../Tiled_images/sar")

    # TODO: read in train files from csv somewhere
    train_sar_files = ["SAR_AP_20181202_00050_[9600,256]_256x256.tiff", "SAR_AP_20181202_00051_[9728,256]_256x256.tiff"]
    train_chart_files = ["BINARY_CHART_AP_20181202_00050_[9600,256]_256x256.tiff", "BINARY_CHART_AP_20181202_00051_[9728,256]_256x256.tiff"]
    train_dataset = SeaIceDataset(sar_path=str(sar_folder),
                                  sar_files=train_sar_files,
                                  chart_path=str(chart_folder),
                                  chart_files=train_chart_files,
                                  transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)

    # TODO: read in val files from csv somewhere
    val_sar_files = ["SAR_AP_20181202_00052_[9856,256]_256x256.tiff", "SAR_AP_20181202_00053_[9984,256]_256x256.tiff"]
    val_chart_files = ["BINARY_CHART_AP_20181202_00052_[9856,256]_256x256.tiff", "BINARY_CHART_AP_20181202_00053_[9984,256]_256x256.tiff"]
    val_dataset = SeaIceDataset(sar_path=str(sar_folder),
                                sar_files=val_sar_files,
                                chart_path=str(chart_folder),
                                chart_files=val_chart_files,
                                transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # TODO: read in test files from csv somewhere
    test_sar_files = ["SAR_AP_20181202_00073_[9344,384]_256x256.tiff", "SAR_AP_20181202_00074_[9472,384]_256x256.tiff"]
    test_chart_files = ["BINARY_CHART_AP_20181202_00073_[9344,384]_256x256.tiff", "BINARY_CHART_AP_20181202_00074_[9472,384]_256x256.tiff"]
    test_dataset = SeaIceDataset(sar_path=str(sar_folder),
                                 sar_files=test_sar_files,
                                 chart_path=str(chart_folder),
                                 chart_files=test_chart_files,
                                 transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # configure model
    if args.model == "unet":
        model = UNet(kernel=3, n_channels=3, n_filters=args.n_filters, n_classes=2)
    else:
        raise ValueError("Unsupported model type")
    criterion = nn.CrossEntropyLoss()
    metric = JaccardIndex(task="binary")
    segmenter = Segmentation(train_dataloader, val_dataloader, model, criterion, args.learning_rate, metric)

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

    # train model
    trainer.fit(segmenter, train_dataloader, val_dataloader)
