import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Segmentation(pl.LightningModule):
    """
    A LightningModule designed to perform image segmentation.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: callable,
                 learning_rate: float,
                 metric: callable,
                 ):
        """
        Construct a Segmentation LightningModule.
        Note that we keep hyperparameters separate from dataloaders to prevent data leakage at test time.
        :param model: PyTorch model
        :param criterion: PyTorch loss function against which to train model
        :param learning_rate: Float learning rate for our optimiser
        :param metric: PyTorch function for model evaluation
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.metric = metric
        self.save_hyperparameters()

    def forward(self, x):
        """
        Implement forward function.
        :param x: Inputs to model.
        :return: Outputs of model.
        """
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Perform a pass through a batch of training data.
        :param batch: Batch of image pairs
        :param batch_idx: Index of batch
        :return: Loss from this batch of data for use in backprop
        """
        x, y = batch["sar"], batch["chart"].squeeze().long()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["sar"], batch["chart"].squeeze().long()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        y_hat_pred = y_hat.argmax(dim=1)
        metric = self.metric(y_hat_pred, y)
        self.log("val_loss", loss)
        self.log("val_metric", metric)
        return loss

    def testing_step(self, batch, batch_idx):
        x, y = batch["sar"], batch["chart"].squeeze().long()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        y_hat_pred = y_hat.argmax(dim=1)
        metric = self.metric(y_hat_pred, y)
        self.log("test_loss", loss)
        self.log("test_metric", metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer
        }


class UNet(nn.Module):
    """
    CNN with skip connections (U-Net) for image segmentation (pixel-wise classification).
    """
    def __init__(self, kernel: int, n_channels: int, n_filters: int, n_classes: int):
        """
        Construct a UNet.
        See following graphical diagram.
        x --a------------------a--> x_5
           x_1 --b-------b--> x_4
                x_2 --> x_3
        :param kernel: Convolutional filter size
        :param n_channels: Number of channels in input image
        :param n_filters: Number of convolutional filters to apply in each convolutional layer
        :param n_classes: Number of possible classes for output pixels
        """
        super().__init__()
        stride = 2  # how far to slide the convolutional filter on each step
        padding = kernel // 2  # how much to pad the image on edges of input
        output_padding = 1  # how much to pad the image on edges of output
        blocks = {}  # a dictionary to store the layers of our network as we build it

        # sequentially construct blocks
        # name is the name of our layer in the blocks dictionary
        # c_in is the number of input channels for each convolutional layer
        # c_out is the number of output channels for each convolutional layer
        # down is a boolean telling us if we're on the downsampling (conv) part of our network or upsampling (convT) part of our network
        for name, c_in, c_out, down in [("in_a", n_channels, n_filters, True),
                                        ("in_b", n_filters, 2 * n_filters, True),
                                        ("out_b", 2 * n_filters, n_filters, False),
                                        ("out_a", 2 * n_filters, n_classes, False)]:
            block = []
            if down:  # construct a convolutional layer that downsamples (stride 2) our image
                block.append(nn.Conv2d(in_channels=c_in,
                                       out_channels=c_out,
                                       kernel_size=kernel,
                                       padding=padding,
                                       stride=stride))
            else:  # construct a transposed convolutional layer that upsamples (stride 2) our image
                block.append(nn.ConvTranspose2d(in_channels=c_in,
                                                out_channels=c_out,
                                                kernel_size=kernel,
                                                padding=padding,
                                                stride=stride,
                                                output_padding=output_padding))
            block.append(nn.BatchNorm2d(num_features=c_out))  # append a batch normalisation module to this block
            block.append(nn.ReLU())  # append a relu activation to this block
            blocks[name] = nn.Sequential(*block)  # make this block a proper sequential unit
        blocks["out"] = nn.Conv2d(in_channels=n_channels + n_classes, # add final conv2d to synthesise output
                                  out_channels=n_classes,
                                  kernel_size=kernel,
                                  padding=padding)
        self.model = nn.ModuleDict(blocks)  # construct sequential model from layers

    def forward(self, x):
        """
        Implement forward step through network.
        See following graphical diagram.
        x --+------------------+--> x_5
           x_1 --+-------+--> x_4
                x_2 --> x_3

        :param x: [Tensor] Input of shape [batch_size, num_in_channels, h, w] from CIFAR10.
        :return: [Tensor] Output of shape [batch_size, num_colours, h, w] with colorization classifications.
        """
        x_1 = self.model["in_a"].forward(x)
        x_2 = self.model["in_b"].forward(x_1)
        x_3 = self.model["out_b"].forward(x_2)
        x_4 = self.model["out_a"].forward(torch.cat((x_3, x_1), dim=1))     # skip connection along channel dimension
        x_5 = self.model["out"].forward(torch.cat((x_4, x), dim=1))         # skip connection along channel dimension
        return x_5
