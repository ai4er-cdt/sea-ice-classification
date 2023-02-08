import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset
from model import Segmentation, UNet
from torchmetrics import IoU  # may be called Jaccard Index in newer versions of torchmetrics


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Sea Ice Segmentation")
    parser.add_argument("--name", default="seaice", type=str, help="Name of wandb run")
    parser.add_argument('--gpu_id', default=-1, type=int, help="GPU id to train on")
    parser.add_argument('--n_filters', default=16, type=float, help="Number of convolutional filters in hidden layer")
    parser.add_argument('--learning_rate', default=0.05, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--n_epochs', default=15, type=int, help="Number of epochs to fine-tune")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    args = parser.parse_args()

    if (args.gpu_id == -1):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    # configure dataloaders
    train_dataset = SeaIceDataset(sar_path="./sar", chart_path="./chart", transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataset = SeaIceDataset(sar_path="./sar", chart_path="./chart", transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_dataset = SeaIceDataset(sar_path="./sar", chart_path="./chart", transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # configure model
    if args.model == "unet":
        model = UNet(kernel=3, n_channels=3, n_filters=args.n_filters, n_classes=2)
    else:
        raise ValueError("Unsupported model type")
    criterion = nn.CrossEntropyLoss()
    metric = IoU(num_classes=2)
    segmenter = Segmentation(train_dataloader, val_dataloader, model, criterion, args.learning_rate, metric)

    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="sea-ice-classification")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(args)

    # set up trainer configuration
    trainer = pl.Trainer()
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="val_loss"))

    # train model
    trainer.fit(segmenter, train_dataloader, val_dataloader)
