# 2025 - copyright - all rights reserved - clayton thomas baber

from torch import round, abs
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import Trainer, LightningModule
from dataset import RubikDistanceDataModule

class RubikDistancePredictor(LightningModule):
    def __init__(self, hidden_dim=256, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.network = Sequential(
        Linear(324, hidden_dim),  # Input layer to first hidden layer
        ReLU(),                   # Activation function
        Linear(hidden_dim, hidden_dim // 2), # Second hidden layer
        ReLU(),
        Linear(hidden_dim // 2, 1) # Output layer (single neuron for distance)
        # No final activation (like Sigmoid or Softmax) here,
        # as we'll use MSELoss and round/clamp the continuous output.
        
        )
        self.loss_fn = MSELoss()

    def forward(self, x):
        # Ensure input is float, as nn.Linear expects float
        x = x.float()
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x).squeeze() # Squeeze to match target shape (batch_size,)
        loss = self.loss_fn(predictions, y.float()) # Ensure target is float for MSELoss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x).squeeze()
        val_loss = self.loss_fn(predictions, y.float())
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        # Optional: Log a more "interpretable" metric like Mean Absolute Error (MAE)
        # on the rounded integer predictions
        predicted_distances_rounded = round(predictions).clamp(0, 20).int()
        true_distances_int = y.int() # Assuming y contains integers 0-20

        mae = abs(predicted_distances_rounded - true_distances_int).float().mean()
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)

        return val_loss
    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)

        # Define the learning rate scheduler
        # MultiStepLR will lower the LR at specified 'milestones'
        # Here, it will drop the LR by a factor of 0.1 (gamma) when the step count reaches 70000
        scheduler = MultiStepLR(
            optimizer,
            milestones=[24795, 74385],  # The step at which the learning rate will be reduced
            gamma=0.8            # The factor by which the learning rate will be multiplied
        )

        # Return the optimizer and the scheduler in PyTorch Lightning's required format
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Crucial: update LR after each optimization step (batch)
                'frequency': 1,      # Apply the scheduler every step
                'name': 'lr_scheduler' # Optional: name for logging in TensorBoard
            }
        }

if __name__ == "__main__":
    model = RubikDistancePredictor(512, 1e-3)
    datamodule = RubikDistanceDataModule(1024, 24795)
    trainer = Trainer(max_epochs=7000, benchmark=True, accelerator="gpu")#, overfit_batches=1)
    trainer.fit(model, datamodule)
