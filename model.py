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
from dataset import RubikDataModule, DatasetBuilder, ArchiveProcessor
from cube import Cube
import numpy as np
import datetime
import os
import math
import json

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
      warmup_power=0.5, # < 1.0 accelerates early warmup ramp
      decay_power=0.35, # < 1.0 holds high LR longer for a broader shoulder
      consistency_delay_epochs=20,
      total_epochs=180,
      ramp_end_epoch=120,
      target_ratio_pct=0.05,
      ema_epoch_fraction=0.3,
      k_rotations=8,
      gamma_start=0.35,
      gamma_end=0.10,
      gamma_delay_epochs=20,
      gamma_ramp_end_epoch=120,
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
        warmup_power=self.hparams.warmup_power,
        decay_power=self.hparams.decay_power,
    )

    # Initialize dynamic consistency scheduler
    self.consistency_schedule = AdaptiveSigmoidalConsistencySchedule(
        consistency_delay_epochs=self.hparams.consistency_delay_epochs,
        ramp_end_epoch=self.hparams.ramp_end_epoch,
        target_ratio_pct=self.hparams.target_ratio_pct,
        ema_epoch_fraction=self.hparams.ema_epoch_fraction,
        steps_per_epoch=self.steps_per_epoch,
    )

    # Initialize scheduled gamma blending for consistency target
    self.gamma_schedule = LinearScheduledValue(
        start_val=self.hparams.gamma_start,
        end_val=self.hparams.gamma_end,
        delay_epochs=self.hparams.gamma_delay_epochs,
        ramp_end_epoch=self.hparams.gamma_ramp_end_epoch,
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

  def _apply_rotations(self, x, k_sub=None):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    batch_size = x.size(0)
    
    # Choose subset of K rotations if specified, otherwise use all 24
    if k_sub is not None and k_sub < 24:
      rand_indices = torch.randperm(24, device=x.device)[:k_sub]
      selected_perms = self.rotation_perms[rand_indices]
      num_rotations = k_sub
    else:
      selected_perms = self.rotation_perms
      num_rotations = 24

    x_reshaped = x.view(batch_size, 54, 6)
    x_expanded = x_reshaped.unsqueeze(1).expand(-1, num_rotations, -1, -1)
    idx = selected_perms.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 6)
    x_aug = torch.gather(x_expanded, 2, idx)
    return x_aug.reshape(-1, 324), num_rotations

  def _get_current_consistency_weight(self):
    current_epoch = self.global_step / self.steps_per_epoch
    return self.consistency_schedule.get_weight(current_epoch)

  def _get_current_gamma(self):
    current_epoch = self.global_step / self.steps_per_epoch
    return self.gamma_schedule.get_value(current_epoch)

  def forward(self, x, return_aug=False, k_sub=None):
    c_weight = self._get_current_consistency_weight()

    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Training logic uses subset of K rotations
    if self.training or return_aug:
      x_all, num_rotations = self._apply_rotations(x, k_sub=k_sub)
      logits_all = self.network(x_all)
      if return_aug:
        return logits_all.view(-1, num_rotations, self.hparams.num_classes)
      return logits_all

    # Inference/Validation logic with full 24-rotation consensus
    with torch.no_grad():
      if not self.hparams.augment or c_weight == 0.0:
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)

      all_avg_probs = []
      chunk_size = 128
      for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i : i + chunk_size]
        x_all, _ = self._apply_rotations(x_chunk, k_sub=None) # Full 24 rotations
        logits_all = self.network(x_all)
        logits_reshaped = logits_all.view(-1, 24, self.hparams.num_classes)
        avg_probs = torch.softmax(logits_reshaped, dim=-1).mean(dim=1)
        all_avg_probs.append(avg_probs)
      
      return torch.cat(all_avg_probs, dim=0)

  def training_step(self, batch, batch_idx):
    x, y = batch
    c_weight = self._get_current_consistency_weight()
    gamma = self._get_current_gamma()

    # Convert soft / density / one-hot targets to class indices for CrossEntropy
    if y.dim() > 1:
      y_labels = y.argmax(dim=-1).long()
    else:
      y_labels = y.long()

    if self.hparams.augment:
      k_sub = self.hparams.k_rotations
      logits_aug = self.forward(x, return_aug=True, k_sub=k_sub)  # (B, K, C)
      log_probs = F.log_softmax(logits_aug, dim=-1)               # (B, K, C)

      # 1. Supervision Loss
      y_expanded = y_labels.unsqueeze(1).repeat(1, k_sub).view(-1).long()
      sup_loss = F.cross_entropy(
          logits_aug.view(-1, self.hparams.num_classes),
          y_expanded,
          weight=self.loss_weights,
      )

      # 2. Compute Target Probabilities and Blended Target
      with torch.no_grad():
        sub_centroid = torch.softmax(logits_aug, dim=-1).mean(dim=1).detach() # (B, C)
        y_onehot = F.one_hot(y_labels, num_classes=self.hparams.num_classes).float() # (B, C)
        
        # Blend sub-centroid ensemble with ground truth target
        blended_target = (1.0 - gamma) * sub_centroid + gamma * y_onehot
        blended_target = blended_target.clamp(min=1e-7)

      # 3. Flatten tensors to (B * K, C) to ensure correct batchmean KL divergence scaling
      log_probs_flat = log_probs.view(-1, self.hparams.num_classes)
      centroid_flat = sub_centroid.unsqueeze(1).repeat(1, k_sub, 1).view(-1, self.hparams.num_classes)
      blended_flat = blended_target.unsqueeze(1).repeat(1, k_sub, 1).view(-1, self.hparams.num_classes)

      raw_consistency_loss = F.kl_div(log_probs_flat, centroid_flat, reduction='batchmean')
      blended_consistency_loss = F.kl_div(log_probs_flat, blended_flat, reduction='batchmean')

      # Record blended loss for EMA calculation
      self.consistency_schedule.record_losses(
          sup_loss=sup_loss.item(), raw_kl_loss=blended_consistency_loss.item()
      )

      # 4. Dynamic Weighting with HARD CAP (Guaranteeing target ratio is never exceeded)
      unclipped_weighted_consistency = c_weight * blended_consistency_loss
      max_allowed_consistency = (sup_loss + 1e-8) * self.hparams.target_ratio_pct
      
      weighted_consistency = torch.minimum(
          unclipped_weighted_consistency, max_allowed_consistency
      )
      
      loss = sup_loss + weighted_consistency

      # 5. Compute Loss Ratio (%)
      loss_ratio = (weighted_consistency / (sup_loss + 1e-8)) * 100.0

      # Logging
      self.log('loss/sup_loss', sup_loss, on_step=True, on_epoch=True, prog_bar=False)
      self.log('loss/consistency_loss_raw', raw_consistency_loss, on_step=True, on_epoch=True, prog_bar=False)
      self.log('loss/consistency_loss_blended', blended_consistency_loss, on_step=True, on_epoch=True, prog_bar=False)
      self.log('loss/weighted_consistency', weighted_consistency, on_step=True, on_epoch=True, prog_bar=False)
      self.log('loss/consistency_ratio_pct', loss_ratio, on_step=True, on_epoch=True, prog_bar=False)
      self.log('params/gamma', gamma, on_step=True, prog_bar=False)
    else:
      logits = self.network(x)
      loss = F.cross_entropy(logits, y_labels)

    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('c_weight', c_weight, on_step=True, prog_bar=False)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch

    if y.dim() > 1:
      y_long = y.argmax(dim=-1).long()
      y_scalar = y.argmax(dim=-1).float()
    else:
      y_long = torch.clamp(y.long(), 0, self.hparams.num_classes - 1)
      y_scalar = y.float()
    
    probs = self.forward(x) 
    
    val_loss = F.nll_loss(torch.log(probs + 1e-9), y_long)
    
    preds = torch.argmax(probs, dim=-1)
    acc = (preds == y_long).float().mean()
    
    distances = torch.arange(self.hparams.num_classes, device=self.device).float()
    expected_distances = (probs * distances).sum(dim=-1)
    ev_error = torch.abs(expected_distances - y_scalar).mean()
    
    self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
    self.log('val_acc', acc, on_epoch=True, prog_bar=False)
    self.log('val_ev_error', ev_error, on_epoch=True, prog_bar=False)

    return val_loss

  def configure_optimizers(self):
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
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5

    was_clipped = 1.0 if total_norm > self.hparams.grad_clip else 0.0
    
    self.clipping_history.append(was_clipped)
    if len(self.clipping_history) > self.window_size:
        self.clipping_history.pop(0)
    
    clipping_freq = sum(self.clipping_history) / len(self.clipping_history)
    
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
      warmup_power: float = 0.5,
      decay_power: float = 0.35,
  ):
    self.log_start_lr = math.log10(start_lr)
    self.log_peak_lr = math.log10(peak_lr)
    self.log_end_lr = math.log10(end_lr)
    self.warmup_steps = int(warmup_epochs * steps_per_epoch)
    self.total_steps = int(total_epochs * steps_per_epoch)
    self.decay_steps = max(1, self.total_steps - self.warmup_steps)
    self.warmup_power = warmup_power
    self.decay_power = decay_power

  def get_value(self, current_step: int) -> float:
    if current_step < self.warmup_steps:
      progress = current_step / max(1, self.warmup_steps)
      # Shape the warmup using warmup_power exponent (p)
      progress = progress ** self.warmup_power
      log_lr = self.log_start_lr + (self.log_peak_lr - self.log_start_lr) * math.sin(0.5 * math.pi * progress)
    else:
      progress = (current_step - self.warmup_steps) / self.decay_steps
      progress = min(max(progress, 0.0), 1.0)
      # Shape the cosine decay using decay_power exponent (q)
      cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
      shaped_decay = cosine_decay ** self.decay_power
      log_lr = self.log_end_lr + (self.log_peak_lr - self.log_end_lr) * shaped_decay
    
    return 10.0 ** log_lr


class AdaptiveSigmoidalConsistencySchedule:
  """Two-phase consistency scheduler: Centroid Alignment Delay -> Smooth Sigmoidal Ramp."""

  def __init__(
      self,
      consistency_delay_epochs=30,
      ramp_end_epoch=120,
      target_ratio_pct=0.05,
      ema_epoch_fraction=0.3,
      steps_per_epoch=36506,
  ):
    self.delay_epochs = consistency_delay_epochs
    self.ramp_end_epoch = ramp_end_epoch
    self.target_ratio_pct = target_ratio_pct

    steps_in_fraction = max(1.0, ema_epoch_fraction * steps_per_epoch)
    self.alpha = 1.0 - math.exp(-1.0 / steps_in_fraction)

    self.ema_sup = None
    self.ema_kl = None

  def record_losses(self, sup_loss: float, raw_kl_loss: float):
    if self.ema_sup is None:
      self.ema_sup = float(sup_loss)
      self.ema_kl = float(raw_kl_loss)
    else:
      self.ema_sup = self.alpha * float(sup_loss) + (1.0 - self.alpha) * self.ema_sup
      self.ema_kl = self.alpha * float(raw_kl_loss) + (1.0 - self.alpha) * self.ema_kl

  def get_weight(self, current_epoch: float) -> float:
    if self.ema_sup is None or self.ema_kl is None or self.ema_kl <= 1e-8:
      return 0.0

    if current_epoch < self.delay_epochs:
      return 0.0

    if current_epoch < self.ramp_end_epoch:
      tau = (current_epoch - self.delay_epochs) / max(1e-5, (self.ramp_end_epoch - self.delay_epochs))
      tau = min(max(tau, 0.0), 1.0)
      smooth_factor = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
      desired_pct = self.target_ratio_pct * smooth_factor
    else:
      desired_pct = self.target_ratio_pct

    base_factor = desired_pct / (1.0 - desired_pct)
    raw_weight = base_factor * (self.ema_sup / self.ema_kl)
    return raw_weight

class LinearScheduledValue:
  """Schedules a parameter value linearly from start_val to end_val across specified epochs."""
  def __init__(self, start_val: float, end_val: float, delay_epochs: float, ramp_end_epoch: float):
    self.start_val = start_val
    self.end_val = end_val
    self.delay_epochs = delay_epochs
    self.ramp_end_epoch = ramp_end_epoch

  def get_value(self, current_epoch: float) -> float:
    if current_epoch < self.delay_epochs:
      return self.start_val
    if current_epoch >= self.ramp_end_epoch:
      return self.end_val
    progress = (current_epoch - self.delay_epochs) / max(1e-5, (self.ramp_end_epoch - self.delay_epochs))
    return self.start_val + progress * (self.end_val - self.start_val)


if __name__ == "__main__":  
  import signal
  def manual_skip_handler(signum, frame):
      raise ValueError("Manual skip signal received.")

  signal.signal(signal.SIGUSR1, manual_skip_handler)

  ARCHIVE_PATH = "assets/htm4.zip"
  MASTER_DIR = "data/master_archive_data"
  
  # Ensure base archive processing and dataset construction exist
  if not os.path.exists(os.path.join(MASTER_DIR, "master_meta.json")):
    ArchiveProcessor.process_archive(ARCHIVE_PATH, output_dir=MASTER_DIR, chunk_size=50000)

  for run in range(6):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_run_{run}"
    run_dir = os.path.join("checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)

    builder = DatasetBuilder(MASTER_DIR)
    builder.build_dataset(
      output_dir=run_dir,
      cycle_filter=2,
      num_select=16,
      max_shared_prefix=3,
      shift_offsets=[0, 4, 8, 12, 16],
      transform_indices=list(range(48)),
      off_path_penalty=0.1,
      branch_depth_k=2,
      random_seed=None
    )


    train_batch_size = 768
    with open(os.path.join(run_dir, "dataset_meta.json"), "r") as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]

    datamodule = RubikDataModule(
        data_dir=run_dir,
        input_type="onehot",
        target_type="onehot",
        batch_size=train_batch_size,
        num_workers=4,
        num_classes=num_classes,
        enable_online_rotations=False
    )
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

    model = RubikDistancePredictor(
        hidden_dim=4096,
        train_ds_size=len(datamodule.train_ds),
        batch_size=datamodule.batch_size,
        start_lr=3e-5,
        peak_lr=1.2e-3,
        end_lr=1e-6,
        warmup_epochs=10,
        warmup_power=0.5,
        decay_power=1.25,
        consistency_delay_epochs=0,
        total_epochs=180,
        ramp_end_epoch=60,
        target_ratio_pct=0.08,
        ema_epoch_fraction=0.3,
        k_rotations=8,
        gamma_start=0.35,
        gamma_end=0.10,
        gamma_delay_epochs=30,
        gamma_ramp_end_epoch=120,
        augment=True,
        num_classes=num_classes,
        class_weights=datamodule.class_weights,
        grad_clip=5.0,
    )

    trainer = Trainer(
      max_epochs=model.total_epochs,
      benchmark=True,
      accelerator="gpu",
      logger=tb_logger,
      callbacks=[lr_monitor, checkpoint_callback],
      precision="16-mixed",
      gradient_clip_val=model.hparams.grad_clip,
      val_check_interval=0.5
    )

    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"Skipping Current Run: {e}")
        empty_cache()
        continue