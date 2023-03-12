"""
AI4ER GTC - Sea Ice Classification
Script for feeding test data into unet or 
resnet34 model and saving the model output to wandb
"""
import os
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from constants import new_classes
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from util import SeaIceDataset
from model import Segmentation
from torchmetrics import JaccardIndex  
from pathlib import Path


if __name__ == "__main__":

    parser = ArgumentParser(description="Sea Ice Segmentation Test")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--username", type=str, help="wandb username")
    parser.add_argument("--name", type=str, help="Name of wandb run")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--overfit", default=False, type=eval, help="Whether or not to overfit on a single image")
    parser.add_argument("--classification_type", default=None, type=str, help="[binary,ternary,multiclass]")
    parser.add_argument("--n_workers", default=1, type=eval, help="Number of subprocesses for data loading")
    args = parser.parse_args()

    # standard input dirs
    base_folder = "../Tiled_images"
    sar_folder = f"{base_folder}/sar"
    chart_folder = f"{base_folder}/binary_chart"
    test_folder = f"{base_folder}/test"

    # get test file list
    if args.overfit:  # load single train/val/test file and overfit
        print("overfitting...")
        test_files = ["AP_20181202_00040_[9216,512]_256x256.tiff", "AP_20181202_00040_[9216,512]_256x256.tiff"]
    else:  
        with open(Path(f"{test_folder}/test_files.txt"), "r") as f:
            test_files = f.read().splitlines()
    
    # init
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)

    # load test data
    test_sar_files = [f"SAR_{f}" for f in test_files]
    test_chart_files = [f"CHART_{f}" for f in test_files]
    test_dataset = SeaIceDataset(sar_path=sar_folder,sar_files=test_sar_files,
                                 chart_path=chart_folder,chart_files=test_chart_files,
                                 class_categories=class_categories)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # wandb logging
    wandb.init(id=args.name, project="sea-ice-classification", resume="must")
    api = wandb.Api()
    run = api.run(f"{args.username}/sea-ice-classification/{args.name}")

    # load model from best checkpoint
    checkpoint_folder = Path(f"./sea-ice-classification/{args.name}/checkpoints")
    checkpoints = os.listdir(checkpoint_folder)
    best_epoch = 0
    best_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("=")[1].split("-")[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_checkpoint = f"{checkpoint_folder}/{checkpoint}"
    model = Segmentation.load_from_checkpoint(best_checkpoint)

    # test
    model.eval()
    criterion = model.criterion
    metric = JaccardIndex(task="multiclass")
    losses = []
    metrics = []
    for batch in test_dataloader:
        x, y = batch["sar"], batch["chart"].squeeze().long()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        y_hat_pred = y_hat.argmax(dim=1)
        metric_value = metric(y_hat_pred, y)
        losses.append(loss)
        metrics.append(metric_value)

    # save and log test nll
    test_loss = sum(losses) / len(test_dataset)
    test_metric = sum(losses) / len(test_dataset)
    wandb.log({"test_loss": test_loss})
    wandb.log({"test_metric": test_metric})
