import os
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset
from model import Segmentation, UNet
from torchmetrics import IoU  # may be called Jaccard Index in newer versions of torchmetrics


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--username", default="", type=str, help="wandb username")
    parser.add_argument("--name", default="", type=str, help="Name of wandb run")
    args = parser.parse_args()

    # wandb logging
    wandb.init(id=args.name, project="sea-ice-classification", resume="must")
    api = wandb.Api()
    run = api.run(f"{args.username}/sea-ice-classification/{args.name}")

    # configure dataloaders
    test_dataset = SeaIceDataset(sar_path="./sar", chart_path="./chart", transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # load model from best checkpoint
    model = None
    checkpoint_folder = f"./sea-ice-classification/{args.name}/checkpoints"
    checkpoints = os.listdir(checkpoint_folder)
    best_epoch = 0
    best_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("=")[1].split("-")[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_checkpoint = f"{checkpoint_folder}/{checkpoint}"

    if run.config["model"] == "unet":
        model = UNet.load_from_checkpoint(best_checkpoint)
    else:
        raise ValueError("Unsupported model type")

    # test
    model.eval()
    losses = []
    metrics = []
    for batch in test_dataloader:

        x, y = batch["sar"], batch["chart"]
        y_hat = model(x)
        loss = model.get_criterion()(y_hat, y)
        metric = IoU(num_classes=2)(y_hat, y)
        losses.append(loss)
        metrics.append(metric)

    # save and log test nll
    test_loss = sum(losses) / len(test_dataset)
    test_metric = sum(losses) / len(test_dataset)
    wandb.log({"test_loss": test_loss})
    wandb.log({"test_metric": test_metric})
