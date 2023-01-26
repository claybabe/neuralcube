# 2023 - copyright - all rights reserved - clayton thomas baber

from torch import argmax
from torch.nn import Linear, Tanh, Sigmoid, Sequential
from torch.nn.functional import mse_loss
from torch.optim import SGD
from pytorch_lightning import Trainer, LightningModule
from dataset import BrownianAntipodalDataModule

class BrownianAntipodalNavigator(LightningModule):
    def __init__(self):
        super().__init__()
        layers = [
                Linear(486, 242),Tanh(),
                Linear(242, 122),Tanh(),
                Linear(122, 62),Tanh(),
                Linear(62, 18), Sigmoid()
             ]
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        correct, total = 0, 0
        x, y = batch
        y_hat = self(x)
        for k, v in enumerate(y_hat):
            total += 1
            if int(argmax(v)) == int(argmax(y[k])):
                correct += 1
        self.log("val_loss", 1 - correct/total)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.0048)

if __name__ == "__main__":
    model = BrownianAntipodalNavigator()
    datamodule = BrownianAntipodalDataModule(20000, 1280, 1, 8,  1000, 10000, 0, 10)
    trainer = Trainer(max_epochs=10, benchmark=True)    
    trainer.fit(model, datamodule)
