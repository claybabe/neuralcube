# 2026 - copyright - all rights reserved - clayton thomas baber

import torch
from torch.nn import Linear, Sequential, ReLU, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import empty_cache
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import grad_norm
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
               augment=False,
               num_classes=21,
               class_weights = None,
               grad_clip=0.5,
               ):
    super().__init__()
    self.save_hyperparameters()

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
      Linear(self.hparams.hidden_dim // 2, self.hparams.num_classes) 
    )
    self.register_buffer('loss_weights', self.hparams.class_weights)
    self.loss_fn = CrossEntropyLoss(weight=self.loss_weights)

    self.register_buffer('all_color_perms', torch.tensor(Cube.color_rotation, dtype=torch.long))

  def forward(self, x):
    logits = self.network(x)
    
    if self.training:
      return logits
    else:
      return torch.softmax(logits, dim=-1)
    
  def training_step(self, batch, batch_idx):
    x, y = batch # x: [batch, 324]
    
    if self.hparams.augment:
        n = 24  # All physical rotations
        batch_size = x.size(0)

        # 1. Expand the inputs and labels: [batch * 24, 54, 6]
        # repeat_interleave ensures: [Sample1, Sample1... (24x), Sample2, Sample2... (24x)]
        x_aug = x.view(-1, 54, 6).repeat_interleave(n, dim=0)
        y = y.repeat_interleave(n, dim=0)

        # 2. Prepare the permutation indices for the expanded batch
        # This repeats the 24-perm block for every sample in the batch
        # Shape: [batch * 24, 6]
        selected_perms = self.all_color_perms.repeat(batch_size, 1)
        
        # 3. Create the gather index: [batch * 24, 54, 6]
        # We need to expand [batch*24, 6] -> [batch*24, 54, 6]
        gather_idx = selected_perms.unsqueeze(1).expand(-1, 54, -1)

        # 4. Apply the rotation across the color dimension and flatten
        x = torch.gather(x_aug, 2, gather_idx).reshape(-1, 324)

    logits = self(x)
    loss = self.loss_fn(logits, y.long())
    
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y = torch.clamp(y.long(), 0, self.hparams.num_classes - 1)
    
    # Get raw logits for loss, then softmax for metrics
    logits = self.network(x)
    probs = torch.softmax(logits, dim=-1) # shape: [batch, 21]
    
    # 1. Standard Loss (CrossEntropy)
    val_loss = self.loss_fn(logits, y)
    
    # 2. Hard Accuracy (Did we pick the exact move?)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean()
    
    # 3. Expected Value (EV) Calculation
    # Create a vector of distances: [0, 1, 2, ..., 20]
    distances = torch.arange(self.hparams.num_classes, device=self.device).float()
    
    # Batch dot product: sum(probs * distances)
    expected_distances = (probs * distances).sum(dim=-1)
    
    # EV Error: How far is our "average" guess from the target?
    ev_error = torch.abs(expected_distances - y.float()).mean()
    
    self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
    self.log('val_acc', acc, on_epoch=True, prog_bar=True)
    self.log('val_ev_error', ev_error, on_epoch=True, prog_bar=True)
    
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

  def on_before_optimizer_step(self, optimizer):
    # Calculate and log the L2 norm (2-norm) of gradients
    norms = grad_norm(self, norm_type=2)
    self.log_dict(norms)

    # Calculate the total L2 norm
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    # This ** 0.5 is the Square Root (Math)
    total_norm = total_norm ** 0.5

    # Log the raw norm
    self.log('grad/total_norm', total_norm, on_step=True)
    
    # Log the clip threshold from your hparams (Logic)
    # This ensures your log matches what the Trainer is actually doing
    self.log('grad/clip_threshold', self.hparams.grad_clip, on_step=True)
    
    # Log how many times the norm exceeded the threshold
    was_clipped = 1.0 if total_norm > self.hparams.grad_clip else 0.0
    self.log('grad/was_clipped_flag', was_clipped, on_step=True)

class RubikEnsemble:
  def __init__(self, model_paths, device="cpu"):
    self.models = []
    for path in model_paths:
      m = RubikDistancePredictor.load_from_checkpoint(path, map_location=device, strict=False)
      m.eval()
      self.models.append(m)
    print(f"Ensemble loaded with {len(self.models)} models.")

  def __call__(self, x):
    with torch.no_grad():
      preds = torch.stack([m(x) for m in self.models])
      avg_probs = torch.mean(preds, dim=0)
      #return avg_probs
      #hack until evaluate.py update
      max_probs, argmaxes = torch.max(avg_probs, dim=-1)
      decimal = 1.0 - max_probs
      return argmaxes.float() + decimal

if __name__ == "__main__":
  # 0. generate data
  manager = RubikManager()
  manager.generate_dataset(Cube.orbits, deep_layers=2)
  
  for MAXLR in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:

    # 1. Initialize Datamodule
    datamodule = RubikDistanceDataModule(train_batch_size=256, val_batch_size=24795, train_split=0.98)
    
    # 2. Manual Setup to populate the subsets
    datamodule.setup()

    # 3. Define a Learning Rate Schedule
    start_lr = 0
    schedule_lr = [
      (MAXLR, 10),
      (1e-5, 200)
    ]
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
      monitor='val_acc',        # We want the highest accuracy
      dirpath='checkpoints/',    # Where to save
      filename='rubik-{epoch:02d}-{val_acc:.4f}',
      save_top_k=3,             # Keep the best 3 models
      mode='max',               # 'max' because higher val_acc is better
      save_last=True            # Always keep 'last.ckpt' for easy resuming
    )

    # 4. Initialize Model
    model = RubikDistancePredictor(
      hidden_dim=4096,
      train_ds_size=len(datamodule.train_ds),
      batch_size=datamodule.train_batch_size,
      start_lr=start_lr,
      schedule_lr=schedule_lr,
      augment=True,
      class_weights=datamodule.class_weights,
      grad_clip=1.5
    )

    # 5. Hire a Trainer
    trainer = Trainer(
      max_epochs=model.total_epochs,
      benchmark=True,
      accelerator="gpu",
      callbacks=[lr_monitor, checkpoint_callback],
      precision="16-mixed",
      # gradient_clip_val helps prevent the NaN before it happens
      gradient_clip_val=model.hparams.grad_clip, 
    )
    
    print(f"\n--- Testing MAXLR: {MAXLR} ---")
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        # Catching the exception (like NaN loss or exploding gradients)
        # and moving to the next iteration
        print(f"Skipping MAXLR {MAXLR} due to training instability: {e}")
        
        # Crucial: Clean up GPU memory before starting the next run
        empty_cache()
        continue