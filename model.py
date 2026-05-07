# 2026 - copyright - all rights reserved - clayton thomas baber

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import empty_cache
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
               class_weights=None,
               grad_clip=0.5,
               consistency_weight=None  # Default to None for legacy compatibility
               ):
    super().__init__()
    self.save_hyperparameters()

    self.clipping_history = []
    self.window_size = 100
    
    self.total_epochs = sum(stage[1] for stage in schedule_lr)
    self.steps_per_epoch = int(np.ceil(train_ds_size / batch_size))
    self.total_steps = self.total_epochs * self.steps_per_epoch

    self.network = Sequential(
      Linear(324, self.hparams.hidden_dim), ReLU(),
      Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 2), ReLU(),
      Linear(self.hparams.hidden_dim // 2, self.hparams.num_classes) 
    )
    
    self.register_buffer('loss_weights', self.hparams.class_weights)
    self.loss_fn = CrossEntropyLoss(weight=self.loss_weights)

    # Prepare 24 permutations
    self.register_buffer('rotation_perms', torch.tensor(Cube.rotations, dtype=torch.long))

  def _apply_rotations(self, x):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    batch_size = x.size(0)
    x_reshaped = x.view(batch_size, 54, 6)
    x_expanded = x_reshaped.unsqueeze(1).expand(-1, 24, -1, -1)
    idx = self.rotation_perms.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 6)
    x_aug = torch.gather(x_expanded, 2, idx)
    return x_aug.reshape(-1, 324)

  def forward(self, x, return_aug=False):
    c_weight = self.hparams.get('consistency_weight', None)
    is_legacy = c_weight is None or c_weight == 0

    # Ensure we have a batch dimension [Batch, 324]
    # This prevents simulate.py from breaking if it passes a flat [324] tensor
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Training logic remains the same (handles training batch size)
    if self.training or return_aug:
      x_all = self._apply_rotations(x)
      logits_all = self.network(x_all)
      if return_aug:
        return logits_all.view(-1, 24, self.hparams.num_classes)
      return logits_all

    # Inference/Validation logic with Memory Management
    with torch.no_grad():
      if is_legacy or not self.hparams.augment:
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)

      # For large validation batches, we process one cube at a time or in small chunks
      # to prevent the 24x multiplier from triggering an OOM
      all_avg_probs = []
      
      # Process in small chunks (e.g., 128 cubes at a time)
      chunk_size = 128
      for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i : i + chunk_size]
        x_all = self._apply_rotations(x_chunk)
        logits_all = self.network(x_all)
        logits_reshaped = logits_all.view(-1, 24, self.hparams.num_classes)
        avg_probs = torch.softmax(logits_reshaped, dim=-1).mean(dim=1)
        all_avg_probs.append(avg_probs)
      
      return torch.cat(all_avg_probs, dim=0)

  def training_step(self, batch, batch_idx):
    x, y = batch
    c_weight = self.hparams.get('consistency_weight', 0) or 0
    
    if self.hparams.augment:
      # 1. Forward pass (Batch x 24)
      logits_aug = self.forward(x, return_aug=True)
      
      # 2. Stability Fix: Use Log-Softmax directly
      log_probs = F.log_softmax(logits_aug, dim=-1)
      
      # 3. Supervision Loss: Only calculate on the flattened view
      y_expanded = y.unsqueeze(1).repeat(1, 24).view(-1).long()
      sup_loss = F.cross_entropy(logits_aug.view(-1, self.hparams.num_classes), y_expanded, weight=self.loss_weights)

      # 4. Consistency Loss: Prevent log(0) and NaNs
      with torch.no_grad():
        # We target the average probability across orientations
        target_probs = torch.softmax(logits_aug, dim=-1).mean(dim=1).detach()
        # Add epsilon to prevent log(0) if model is perfectly confident
        target_probs = target_probs.clamp(min=1e-7)

      # KL Div using log_probs we already calculated
      consistency_loss = F.kl_div(
        log_probs, 
        target_probs.unsqueeze(1).expand_as(log_probs), 
        reduction='batchmean'
      )
      
      loss = sup_loss + (c_weight * consistency_loss)
    else:
      # Standard non-augmented path
      logits = self.network(x)
      loss = F.cross_entropy(logits, y.long())

    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)  
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_long = torch.clamp(y.long(), 0, self.hparams.num_classes - 1)
    
    # Validation uses the 'Consensus' forward pass (average of 24 rotations)
    probs = self.forward(x) 
    
    # CrossEntropy from probabilities (using log for safety)
    val_loss = F.nll_loss(torch.log(probs + 1e-9), y_long)
    
    preds = torch.argmax(probs, dim=-1)
    acc = (preds == y_long).float().mean()
    
    distances = torch.arange(self.hparams.num_classes, device=self.device).float()
    expected_distances = (probs * distances).sum(dim=-1)
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
    # Calculate the total L2 norm
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    # This ** 0.5 is the Square Root (Math)
    total_norm = total_norm ** 0.5

    was_clipped = 1.0 if total_norm > self.hparams.grad_clip else 0.0
    
    # Maintain a rolling window for a frequency metric
    self.clipping_history.append(was_clipped)
    if len(self.clipping_history) > self.window_size:
        self.clipping_history.pop(0)
    
    # Log the frequency (0.0 to 1.0)
    clipping_freq = sum(self.clipping_history) / len(self.clipping_history)
    
    # This will now look like a "heat map" of how much the model is struggling
    self.log('grad/clipping_frequency', clipping_freq, on_step=True, prog_bar=False)
    self.log('grad/total_norm', total_norm, on_step=True)

class RubikEnsemble:
  def __init__(self, model_paths, device="cpu"):
    self.device = device
    self.models = []
    for path in model_paths:
      m = RubikDistancePredictor.load_from_checkpoint(path, map_location=device, strict=False)
      m.eval()
      self.models.append(m)
    
    num_classes = self.models[0].hparams.num_classes
    # Create the distance vector: [0.0, 1.0, 2.0, ..., num_classes.0]
    # We use register_buffer or just a tensor here to multiply against the probs
    self.distances = torch.arange(num_classes, device=device).float()
    print(f"Ensemble loaded with {len(self.models)} models.")

  def __call__(self, x):
    with torch.no_grad():
      preds = torch.stack([m(x) for m in self.models])
      avg_probs = torch.mean(preds, dim=0)
      expected_value = (avg_probs * self.distances).sum(dim=-1)
      return expected_value

if __name__ == "__main__":
  # 0. generate data
  manager = RubikManager()
  manager.generate_dataset(Cube.orbits, deep_layers=2)
  
  # --- SIGNAL HANDLER SETUP ---
  import signal
  def manual_skip_handler(signum, frame):
      # This will be caught by your 'except Exception as e' block below
      raise ValueError("Manual skip signal received.")

  # Register the listener
  signal.signal(signal.SIGUSR1, manual_skip_handler)
  # ----------------------------

  for MAXLR in [1.2, 1.1, 1.0, 0.9, 0.8, 0.7]:

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
      #save_top_k=3,             # Keep the best 3 models
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
      grad_clip=5,
      consistency_weight=3,
    )

    # 5. Hire a Trainer
    trainer = Trainer(
      max_epochs=model.total_epochs,
      benchmark=True,
      accelerator="gpu",
      callbacks=[lr_monitor, checkpoint_callback],
      precision="16-mixed",
      gradient_clip_val=model.hparams.grad_clip,
    )
    
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"Skipping Current Run: {e}")
        # Crucial: Clean up GPU memory before starting the next run
        empty_cache()
        continue