# 2025 - copyright - all rights reserved - clayton thomas baber

import torch
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import RubikDistanceDataModule, RubikManager
from cube import Cube
import numpy as np
   
class RubikDistancePredictor(LightningModule):
  def __init__(self,
               hidden_dim=256,
               train_ds_size=23364033,
               batch_size=64,
               start_lr=0.001,
               schedule_lr=[(0.03, 5), (0.02, 2), (0.02, 40), (0.0001, 50)],
               augment=False
               ):
    super().__init__()
    self.save_hyperparameters()
    
    # Timeline derived from the Schedule
    self.total_epochs = sum(stage[1] for stage in schedule_lr)
    self.steps_per_epoch = int(np.ceil(train_ds_size / batch_size))
    self.total_steps = self.total_epochs * self.steps_per_epoch
    
    print(f"--- SCHEDULE INITIALIZED ---")
    print(f"Total Run Time: {self.total_epochs} Epochs")
    print(f"Total Steps:    {self.total_steps}")
    print(f"Starting LR:    {self.hparams.start_lr}")
    print(f"Schedule:       {self.hparams.schedule_lr}")

    self.network = Sequential(
      Linear(324, self.hparams.hidden_dim), ReLU(),
      Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 2), ReLU(),
      Linear(self.hparams.hidden_dim // 2, 1)
    )
    self.loss_fn = MSELoss()

  def forward(self, x):
    return self.network(x)

  def training_step(self, batch, batch_idx):
    x, y = batch # x shape: [batch, 324], y shape: [batch]

    if self.hparams.augment:    
      # 1. Expand each item in the batch 6 times
      # x_expanded shape: [batch * 6, 324]
      x_aug = torch.repeat_interleave(x, 6, dim=0)
      y_aug = torch.repeat_interleave(y, 6, dim=0)
      
      # 2. Reshape to manipulate the 6 color channels
      # [batch * 6, 54, 6]
      x_aug = x_aug.view(-1, 54, 6)
      
      # 3. Apply all 6 cyclic shifts across the augmented dimension
      # We create an index tensor to shift each of the 6 copies differently
      for i in range(6):
        # Every 6th element gets a shift of 'i'
        # This ensures each original 'x' is now represented in all 6 color states
        x_aug[i::6] = torch.roll(x_aug[i::6], shifts=i, dims=2)
      
      # 4. Flatten back for the network
      x_aug = x_aug.reshape(-1, 324)
      x, y = x_aug, y_aug


    # 5. Forward pass on the augmented batch
    predictions = self(x).squeeze()
    loss = self.loss_fn(predictions, y)
    
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    predictions = self(x).squeeze()
    val_loss = self.loss_fn(predictions, y)
    
    # Accuracy: Percentage of predictions within ±0.5 of integer target
    diff = torch.abs(predictions - y)
    acc = (diff < 0.5).float().mean()
    
    self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
    self.log('val_acc', acc, on_epoch=True, prog_bar=True)
    return val_loss

  def configure_optimizers(self):
    # Base LR is 1.0 because the lambda provides absolute LR values
    optimizer = SGD(self.parameters(), lr=1.0)
    
    segments = []
    current_step_boundary = 0
    current_lr = self.hparams.start_lr

    for target_lr, duration_epochs in self.hparams.schedule_lr:
      duration_steps = int(duration_epochs * self.steps_per_epoch)
      segments.append({
        "start": current_step_boundary,
        "end": current_step_boundary + duration_steps,
        "lrs": (current_lr, target_lr)
      })
      current_step_boundary += duration_steps
      current_lr = target_lr

    def schedule_lambda(current_step):
      for seg in segments:
        if seg["start"] <= current_step < seg["end"]:
          # Linear interpolation within segments
          t = (current_step - seg["start"]) / (seg["end"] - seg["start"])
          return seg["lrs"][0] + t * (seg["lrs"][1] - seg["lrs"][0])
      
      # Final floor value if run exceeds steps
      return self.hparams.schedule_lr[-1][0]

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': LambdaLR(optimizer, schedule_lambda),
        'interval': 'step',
        'name': 'lr_scheduler'
      }
    }
  
if __name__ == "__main__":
  # 0. generate data
  manager = RubikManager()
  manager.generate_dataset(Cube.orbits, deep_layers=2)

  # 1. Initialize Datamodule
  datamodule = RubikDistanceDataModule(train_batch_size=64, val_batch_size=24795, train_split=0.98)
  
  # 2. Manual Setup to populate the subsets
  datamodule.setup()

  # 3. Define a Learning Rate Schedule
  start_lr = 0
  schedule_lr = [
    (0.05, 2),    # Aggressive Warmup
    (0.0001, 30)  # Extended Precision Landing
  ]
  lr_monitor = LearningRateMonitor(logging_interval='step')
  
  # 4. Initialize Model
  model = RubikDistancePredictor(
    hidden_dim=4536,
    train_ds_size=len(datamodule.train_ds),
    batch_size=datamodule.train_batch_size,
    start_lr=start_lr,
    schedule_lr=schedule_lr,
    augment=True
  )

  # 5. Hire a Trainer
  trainer = Trainer(
    max_epochs=model.total_epochs,
    benchmark=True,
    accelerator="gpu",
    callbacks=[lr_monitor]
  )
  
  # 6. Hit the gym
  trainer.fit(model, datamodule)