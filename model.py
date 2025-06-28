# 2025 - copyright - all rights reserved - clayton thomas baber

from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import RubikDistanceDataModule

class RubikDistancePredictor(LightningModule):
    def __init__(self,
                 hidden_dim=256,
                 base_lr=1e-3,
                 max_lr=2e-2,
                 step_up=88480,
                 step_down=265440
        ):
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
        optimizer = SGD(self.parameters(), lr=self.hparams.base_lr)
        scheduler = CyclicLR(
            optimizer,
            base_lr = self.hparams.base_lr,
            max_lr = self.hparams.max_lr,
            step_size_up = self.hparams.step_up,
            step_size_down = self.hparams.step_down
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
    for lr in [1e-4, 5e-5, 1e-5]:
        model = RubikDistancePredictor(
            hidden_dim = 6480,
            base_lr = lr,
            max_lr = 2e-2,
            step_up = 265400,
            step_down = 796200
        )
        datamodule = RubikDistanceDataModule(64, 24795, regenerate=True)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            max_epochs=160,
            benchmark=True,
            accelerator="gpu",
            callbacks=[lr_monitor]
        )
        trainer.fit(model, datamodule)
