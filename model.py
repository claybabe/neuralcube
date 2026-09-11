# 2026 - copyright - all rights reserved - clayton thomas baber

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import empty_cache
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import RubikDistanceDataModule, RubikManager, PathDatasetProcessor
from cube import Cube
import numpy as np
import datetime
import os

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
               consistency_weight=None,  # Legacy constant weight
               start_consistency_weight=0.0,
               schedule_consistency=None  # New schedule for consistency weight
               ):
    super().__init__()
    self.save_hyperparameters()

    self.clipping_history = []
    self.window_size = 100
    
    # Calculate total epochs based on the LR schedule
    self.total_epochs = sum(stage[1] for stage in schedule_lr)
    self.steps_per_epoch = int(np.ceil(train_ds_size / batch_size))
    self.total_steps = self.total_epochs * self.steps_per_epoch

    # Initialize scheduling engine for Learning Rate
    self.lr_schedule = PiecewiseSchedule(
      start_val=self.hparams.start_lr,
      schedule=self.hparams.schedule_lr,
      steps_per_epoch=self.steps_per_epoch
    )

    # Initialize scheduling engine for Consistency Weight (if schedule provided)
    if self.hparams.schedule_consistency is not None:
      self.consistency_schedule = PiecewiseSchedule(
        start_val=self.hparams.start_consistency_weight,
        schedule=self.hparams.schedule_consistency,
        steps_per_epoch=self.steps_per_epoch
      )

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

  def _get_current_consistency_weight(self):
    if hasattr(self, 'consistency_schedule'):
      return self.consistency_schedule.get_value(self.global_step)
    c_weight = self.hparams.get('consistency_weight', None)
    return c_weight if c_weight is not None else 0.0

  def forward(self, x, return_aug=False):
    c_weight = self._get_current_consistency_weight()
    is_legacy = (self.hparams.get('consistency_weight', None) is None and 
                 self.hparams.get('schedule_consistency', None) is None) or c_weight == 0

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
    c_weight = self._get_current_consistency_weight()

    if self.hparams.augment:
      logits_aug = self.forward(x, return_aug=True)
      log_probs = F.log_softmax(logits_aug, dim=-1)

      # 1. Supervision Loss
      y_expanded = y.unsqueeze(1).repeat(1, 24).view(-1).long()
      sup_loss = F.cross_entropy(
          logits_aug.view(-1, self.hparams.num_classes),
          y_expanded,
          weight=self.loss_weights,
      )

      # 2. Consistency Loss
      with torch.no_grad():
        target_probs = torch.softmax(logits_aug, dim=-1).mean(dim=1).detach()
        target_probs = target_probs.clamp(min=1e-7)

      consistency_loss = F.kl_div(
          log_probs,
          target_probs.unsqueeze(1).expand_as(log_probs),
          reduction='batchmean',
      )

      # 3. Scaled consistency contribution
      weighted_consistency = c_weight * consistency_loss
      loss = sup_loss + weighted_consistency

      # 4. Compute Loss Ratio (%)
      # Protect against div-by-zero during early initialization
      loss_ratio = (weighted_consistency / (sup_loss + 1e-8)) * 100.0

      # Log individual components to TensorBoard
      self.log('loss/sup_loss', sup_loss, on_step=True, on_epoch=True)
      self.log(
          'loss/consistency_loss_raw',
          consistency_loss,
          on_step=True,
          on_epoch=True,
      )
      self.log(
          'loss/weighted_consistency',
          weighted_consistency,
          on_step=True,
          on_epoch=True,
      )
      self.log(
          'loss/consistency_ratio_pct',
          loss_ratio,
          on_step=True,
          on_epoch=True,
          prog_bar=True,
      )
    else:
      logits = self.network(x)
      loss = F.cross_entropy(logits, y.long())

    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('c_weight', c_weight, on_step=True, prog_bar=True)
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

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': LambdaLR(optimizer, self.lr_schedule.get_value),
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
      print(m.hparams)
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

class PiecewiseSchedule:
  """Generates linear interpolation segments across training steps."""
  def __init__(self, start_val: float, schedule: list, steps_per_epoch: int):
    self.segments = []
    current_step = 0
    current_val = start_val

    for target_val, duration_epochs in schedule:
      duration_steps = int(duration_epochs * steps_per_epoch)
      self.segments.append({
        "start": current_step,
        "end": current_step + duration_steps,
        "vals": (current_val, target_val)
      })
      current_step += duration_steps
      current_val = target_val

    self.fallback_val = schedule[-1][0] if schedule else start_val

  def get_value(self, current_step: int) -> float:
    for seg in self.segments:
      if seg["start"] <= current_step < seg["end"]:
        t = (current_step - seg["start"]) / (seg["end"] - seg["start"])
        return seg["vals"][0] + t * (seg["vals"][1] - seg["vals"][0])

    return self.fallback_val

if __name__ == "__main__":  
  # --- SIGNAL HANDLER SETUP ---
  import signal
  def manual_skip_handler(signum, frame):
      # This will be caught by your 'except Exception as e' block below
      raise ValueError("Manual skip signal received.")

  # Register the listener
  signal.signal(signal.SIGUSR1, manual_skip_handler)
  # ----------------------------

  for run in range(6):

    train_batch_size = 256
    val_batch_size = 24795
    train_split = 0.98

    max_lr = 1.2

    # Create a unique, synchronized run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_run_{run}"
    run_dir = os.path.join("checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)

    pdp = PathDatasetProcessor("assets/htm4.zip", start_idx="random", num_select=32, shift_offsets=[0, 4, 8, 12, 16], max_shared_prefix=5)
    paths = pdp.get_paths()
    endpoints = pdp.get_endpoints()

    manager = RubikManager()
    manager.generate_dataset(paths.tolist(), deep_layers=2)
  
    # Save endpoints directly in the run directory for standalone evaluation
    np.save(os.path.join(run_dir, "endpoints.npy"), endpoints)

    # 1. Initialize Datamodule
    datamodule = RubikDistanceDataModule(train_batch_size=train_batch_size, val_batch_size=val_batch_size, train_split=train_split)
    
    # 2. Manual Setup to populate the subsets
    datamodule.setup()

    # 3. Define Schedules
    start_lr = 0
    schedule_lr = [
      (max_lr, 10),
      (1e-5, 200)
    ]

    start_consistency_weight = 0.0
    schedule_consistency = [
        # --- Warmup Phase ---
        (0.00, 30),  # Epochs 0-30: Pure supervision learning (c_weight = 0)
        # --- Cycle 1 (Peak: 0.08) ---
        (0.08, 12),  # Ramp up to 0.08
        (0.00, 12),  # Ramp down to 0.00
        (0.00, 6),  # Hold at 0.00 for unconstrained exploration
        # --- Cycle 2 (Peak: 0.06) ---
        (0.06, 12),  # Ramp up
        (0.00, 12),  # Ramp down
        (0.00, 6),  # Hold at 0.00
        # --- Cycle 3 (Peak: 0.05) ---
        (0.05, 12),  # Ramp up
        (0.00, 12),  # Ramp down
        (0.00, 6),  # Hold at 0.00
        # --- Cycle 4 (Peak: 0.04) ---
        (0.04, 12),  # Ramp up
        (0.00, 12),  # Ramp down
        (0.00, 6),  # Hold at 0.00
        # --- Cycle 5 (Peak: 0.03) ---
        (0.03, 12),  # Ramp up
        (0.00, 12),  # Ramp down
        (0.00, 6),  # Hold at 0.00
        # --- Cycle 6 (Peak: 0.02) ---
        (0.02, 12),  # Ramp up
        (0.00, 12),  # Ramp down
        (0.00, 6),  # Final hold at 0.00
    ]

    # Synchronized TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="rubik_runs",
        version=run_name
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch:02d}-{val_acc:.4f}",
        monitor='val_acc',
        mode='max',
        save_last=True
    )
    # 4. Initialize Model
    model = RubikDistancePredictor(
      hidden_dim=4096,
      train_ds_size=len(datamodule.train_ds),
      batch_size=datamodule.train_batch_size,
      start_lr=start_lr,
      schedule_lr=schedule_lr,
      start_consistency_weight=start_consistency_weight,
      schedule_consistency=schedule_consistency,
      augment=True,
      class_weights=None,#datamodule.class_weights,
      grad_clip=5,
    )

    # 5. Hire a Trainer
    trainer = Trainer(
      max_epochs=model.total_epochs,
      benchmark=True,
      accelerator="gpu",
      logger=tb_logger,  # <--- Pass synchronized logger here
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