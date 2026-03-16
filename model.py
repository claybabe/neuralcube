# 2025 - copyright - all rights reserved - clayton thomas baber

from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import RubikDistanceDataModule, RubikManager
from cube import Cube
import numpy as np
   
class RubikDistancePredictor(LightningModule):
  def __init__(self,
    hidden_dim = 256,
    max_lr = 0.015,
    total_steps = None,
    epochs = None,
    cycle_momentum = True,
    anneal_strategy = "cos",
    div_factor = 25,
    final_div_factor = 1e4,
    pct_start = 0.3
    ):
    super().__init__()
    self.save_hyperparameters()
    print("RubikDistancePredictor \n", self.hparams)

    self.network = Sequential(
      Linear(324, self.hparams.hidden_dim), ReLU(),
      Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 2), ReLU(),
      Linear(self.hparams.hidden_dim // 2, 1)
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
    optimizer = SGD(self.parameters(), lr=self.hparams.max_lr)
    scheduler = OneCycleLR(
      optimizer,
      max_lr=self.hparams.max_lr,
      total_steps=self.hparams.total_steps,
      epochs=self.hparams.epochs,
      cycle_momentum = self.hparams.cycle_momentum,
      anneal_strategy = self.hparams.anneal_strategy,
      div_factor = self.hparams.div_factor,
      final_div_factor = self.hparams.final_div_factor,
      pct_start = self.hparams.pct_start
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
  # 0. generate data
  manager = RubikManager()
  manager.generate_dataset(Cube.orbits, deep_layers=2)

  # 1. Initialize Datamodule
  datamodule = RubikDistanceDataModule(train_batch_size=64, val_batch_size=24795)
  
  # 2. Manual Setup to populate the subsets
  datamodule.setup()
  
  # 3. Calculate steps properly
  # len(datamodule.train_ds) gives you the count of samples after the 90/10 split
  train_samples = len(datamodule.train_ds)
  batch_size = datamodule.train_batch_size
  max_epochs = 200
  
  # Formula: (total_samples / batch_size) rounded up * epochs
  total_steps = np.ceil(train_samples / batch_size) * max_epochs
  total_steps = int(total_steps) # Schedulers usually want an integer

  lr_monitor = LearningRateMonitor(logging_interval='step')
  for hidden_dim in [7776, 7128, 6480, 5832, 4536]:
    
    trainer = Trainer(
      max_epochs=max_epochs,
      benchmark=True,
      accelerator="gpu",
      callbacks=[lr_monitor]
    )

    model = RubikDistancePredictor(
      hidden_dim = hidden_dim,
      max_lr = 0.02,
      total_steps = total_steps, 
      epochs = trainer.max_epochs,
      #cycle_momentum = False,
      anneal_strategy = "linear",
      div_factor = 2000,
      final_div_factor = 0.0025,
      pct_start = 0.18
    )

    trainer.fit(model, datamodule)