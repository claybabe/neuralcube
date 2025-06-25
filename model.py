# 2025 - copyright - all rights reserved - clayton thomas baber

from torch import round, abs
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import RubikDistanceDataModule

class RubikDistancePredictor(LightningModule):
    def __init__(self, hidden_dim=256, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.network = Sequential(
            Linear(324, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim // 2), ReLU(),
            Linear(hidden_dim // 2, 1)
        )
        self.loss_fn = MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x).squeeze()
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x).squeeze()
        val_loss = self.loss_fn(predictions, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CyclicLR(
            optimizer,
            base_lr = 1e-3,
            max_lr = 2e-2,
            step_size_up = 106160,
            step_size_down = 318480
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'lr_scheduler'
            }
        }
    
if __name__ == "__main__":
    for _ in range(3):
        model = RubikDistancePredictor(5832, 1e-3)
        datamodule = RubikDistanceDataModule(64, 24795)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            max_epochs=128,
            benchmark=True,
            accelerator="gpu",
            callbacks=[lr_monitor]
        )
        trainer.fit(model, datamodule)
