# 2025 - copyright - all rights reserved - clayton thomas baber

from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from dataset import RubikDistanceDataModule, RubikEncoderDataModule
from math import ceil

class RubikEncoder(LightningModule):
    def __init__(self,
        base_lr=5e-4,
        max_lr=2e-2,
        step_up=71534,
        step_down=214602
        ):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        self.loss_fn = MSELoss()

        self.encoder = Sequential(
            Linear(324, 108), ReLU(),
            Linear(108, 54)
        )

        self.decoder = Sequential(
            Linear(54, 108), ReLU(),
            Linear(108, 324)
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        x = batch
        predictions = self(x).squeeze()
        loss = self.loss_fn(predictions, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        predictions = self(x).squeeze()
        val_loss = self.loss_fn(predictions, x)
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

class RubikEncodedDistancePredictor(LightningModule):
    def __init__(self,
                 hidden_dim=1080,
                 base_lr=1e-3,
                 max_lr=2e-2,
                 step_up=88480,
                 step_down=265440,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = RubikEncoder()

        self.network = Sequential(
            Linear(54, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim // 2), ReLU(),
            Linear(hidden_dim // 2, 1)
        )
        self.loss_fn = MSELoss()

    def forward(self, x):
        x = self.encoder.encode(x)
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
    

def modelRPD():

    datamodule = RubikDistanceDataModule(64, 24795)#, regenerate=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    for hidden_dim in [7776, 7128, 6480, 5832, 4536]:
        
        trainer = Trainer(
            max_epochs=200,
            benchmark=True,
            accelerator="gpu",
            callbacks=[lr_monitor]
        )

        model = RubikDistancePredictor(
            hidden_dim = hidden_dim,
            max_lr = 0.02,
            total_steps = ceil(len(datamodule.train_dataset) / datamodule.train_batch) * trainer.max_epochs, 
            epochs = trainer.max_epochs,
            #cycle_momentum = False,
            anneal_strategy = "linear",
            div_factor = 2000,
            final_div_factor = 0.0025,
            pct_start = 0.18
        )

        trainer.fit(model, datamodule)


def modelREPD():

    lr_monitor = LearningRateMonitor(logging_interval='step')
    datamodule = RubikEncoderDataModule(256, 24795)#, regenerate=True)
    encoder = RubikEncoder(
        base_lr = 5e-4,
        max_lr = 2e-2,
        step_up = 71534,
        step_down = 214602
    )
    trainer = Trainer(
        max_epochs=8,
        benchmark=True,
        accelerator="gpu",
        callbacks=[lr_monitor]
    )
    
    trainer.fit(encoder, datamodule)
    #important; 
    #encoder.freeze()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    datamodule = RubikDistanceDataModule(64, 24795, regenerate=True)
        
    model = RubikEncodedDistancePredictor(
        hidden_dim=2048,
        base_lr=1e-3,
        max_lr=1e-2,
        step_up = 176960,
        step_down = 530880
    )
    #important
    model.encoder = encoder

    trainer = Trainer(
        max_epochs=512,
        benchmark=True,
        accelerator="gpu",
        callbacks=[lr_monitor]
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":

    choice = int(input("1) RPD 2) REDP ? "))
    if choice == 1:
        modelRPD()
    elif choice == 2:
        modelREPD()
    else:
        print("Not a Vaild Choice")








    
