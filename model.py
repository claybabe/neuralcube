# 2026 - copyright - all rights reserved - clayton thomas baber

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, CrossEntropyLoss
from torch.optim import AdamW
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
import math

class RubikDistancePredictor(LightningModule):

  def __init__(
      self,
      hidden_dim=256,
      train_ds_size=23364033,
      batch_size=64,
      start_lr=3e-5,
      peak_lr=1.2e-3,
      end_lr=1e-6,
      warmup_epochs=10,
      consistency_delay_epochs=20,
      total_epochs=180,
      ramp_end_epoch=120,
      target_ratio_pct=0.05,
      ema_epoch_fraction=0.3,
      augment=False,
      num_classes=21,
      class_weights=None,
      grad_clip=0.5,
  ):
    super().__init__()
    self.save_hyperparameters()

    self.clipping_history = []
    self.window_size = 100
    
    self.total_epochs = total_epochs
    self.steps_per_epoch = int(np.ceil(train_ds_size / batch_size))
    self.total_steps = self.total_epochs * self.steps_per_epoch

    # Initialize Logarithmic Cosine scheduling engine for Learning Rate
    self.lr_schedule = LogarithmicSingleCycleCosineSchedule(
        start_lr=self.hparams.start_lr,
        peak_lr=self.hparams.peak_lr,
        end_lr=self.hparams.end_lr,
        warmup_epochs=self.hparams.warmup_epochs,
        total_epochs=self.total_epochs,
        steps_per_epoch=self.steps_per_epoch,
    )

    # Initialize dynamic consistency scheduler
    self.consistency_schedule = AdaptiveSigmoidalConsistencySchedule(
        total_epochs=self.total_epochs,
        consistency_delay_epochs=self.hparams.consistency_delay_epochs,
        ramp_end_epoch=self.hparams.ramp_end_epoch,
        target_ratio_pct=self.hparams.target_ratio_pct,
        ema_epoch_fraction=self.hparams.ema_epoch_fraction,
        steps_per_epoch=self.steps_per_epoch,
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
    current_epoch = self.global_step / self.steps_per_epoch
    return self.consistency_schedule.get_weight(current_epoch)

  def forward(self, x, return_aug=False):
    c_weight = self._get_current_consistency_weight()

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
      if not self.hparams.augment or c_weight == 0.0:
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

      # Record losses for EMA calibration in the adaptive scheduler
      self.consistency_schedule.record_losses(
          sup_loss=sup_loss.item(), raw_kl_loss=consistency_loss.item()
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
    optimizer = AdamW(self.parameters(), lr=1.0, weight_decay=1e-4)

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

class LogarithmicSingleCycleCosineSchedule:
  """Quarter-wave log-warmup to peak_lr, followed by continuous log-cosine decay to end_lr."""
  def __init__(
      self,
      start_lr: float,
      peak_lr: float,
      end_lr: float,
      warmup_epochs: int,
      total_epochs: int,
      steps_per_epoch: int,
  ):
    self.log_start_lr = math.log10(start_lr)
    self.log_peak_lr = math.log10(peak_lr)
    self.log_end_lr = math.log10(end_lr)
    self.warmup_steps = int(warmup_epochs * steps_per_epoch)
    self.total_steps = int(total_epochs * steps_per_epoch)
    self.decay_steps = max(1, self.total_steps - self.warmup_steps)

  def get_value(self, current_step: int) -> float:
    if current_step < self.warmup_steps:
      progress = current_step / max(1, self.warmup_steps)
      log_lr = self.log_start_lr + (self.log_peak_lr - self.log_start_lr) * math.sin(0.5 * math.pi * progress)
    else:
      progress = (current_step - self.warmup_steps) / self.decay_steps
      progress = min(max(progress, 0.0), 1.0)
      log_lr = self.log_end_lr + 0.5 * (self.log_peak_lr - self.log_end_lr) * (1.0 + math.cos(math.pi * progress))
    
    return 10.0 ** log_lr

class AdaptiveSigmoidalConsistencySchedule:
  """Three-phase consistency scheduler with late-stage decay: Delay -> Sigmoidal Ramp -> Late-Stage Cosine Decay."""

  def __init__(
      self,
      total_epochs=180,
      consistency_delay_epochs=30,
      ramp_end_epoch=120,
      target_ratio_pct=0.05,
      ema_epoch_fraction=0.3,
      steps_per_epoch=36506,
  ):
    self.total_epochs = total_epochs
    self.delay_epochs = consistency_delay_epochs
    self.ramp_end_epoch = ramp_end_epoch
    self.target_ratio_pct = target_ratio_pct

    steps_in_fraction = max(1.0, ema_epoch_fraction * steps_per_epoch)
    self.alpha = 1.0 - math.exp(-1.0 / steps_in_fraction)

    self.ema_sup = None
    self.ema_kl = None

  def record_losses(self, sup_loss: float, raw_kl_loss: float):
    """Updates exponential moving averages for real-time target weight tracking."""
    if self.ema_sup is None:
      self.ema_sup = float(sup_loss)
      self.ema_kl = float(raw_kl_loss)
    else:
      self.ema_sup = self.alpha * float(sup_loss) + (1.0 - self.alpha) * self.ema_sup
      self.ema_kl = self.alpha * float(raw_kl_loss) + (1.0 - self.alpha) * self.ema_kl

  def get_weight(self, current_epoch: float) -> float:
    """Calculates instantaneous c_weight with zero baseline and smooth late-stage decay."""
    if self.ema_sup is None or self.ema_kl is None or self.ema_kl <= 1e-8:
      return 0.0

    # Phase 0: Centroid Alignment Delay (c_weight = 0.0)
    if current_epoch < self.delay_epochs:
      return 0.0

    # Phase 1: Smooth Sigmoidal (Hermite smoothstep) Ramp (0.0 to target_ratio_pct)
    if current_epoch < self.ramp_end_epoch:
      tau = (current_epoch - self.delay_epochs) / max(1e-5, (self.ramp_end_epoch - self.delay_epochs))
      tau = min(max(tau, 0.0), 1.0)
      smooth_factor = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
      desired_pct = self.target_ratio_pct * smooth_factor
    else:
      desired_pct = self.target_ratio_pct

    # Late-Stage Damping Factor: Soft cosine decay after ramp_end_epoch
    # Drops consistency smoothly in final 33% of run to avoid late ratio blowup
    if current_epoch > self.ramp_end_epoch:
      decay_progress = (current_epoch - self.ramp_end_epoch) / max(1e-5, (self.total_epochs - self.ramp_end_epoch))
      decay_progress = min(max(decay_progress, 0.0), 1.0)
      damping = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    else:
      damping = 1.0

    base_factor = desired_pct / (1.0 - desired_pct)
    raw_weight = base_factor * (self.ema_sup / self.ema_kl) * damping
    return raw_weight

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

    # Create a unique, synchronized run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_run_{run}"
    run_dir = os.path.join("checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_batch_size = 256
    val_batch_size = 24795
    train_split = 0.98

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
        start_lr=3e-5,
        peak_lr=1.2e-3,
        end_lr=1e-6,
        warmup_epochs=10,
        consistency_delay_epochs=30,
        total_epochs=180,
        ramp_end_epoch=120,
        target_ratio_pct=0.05,
        ema_epoch_fraction=0.3,
        augment=True,
        class_weights=None,#datamodule.class_weights,
        grad_clip=5.0,
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