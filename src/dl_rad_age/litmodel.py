# Custom PyTorch Lightning LightningModule, an intelligent wrapper for training in PyTorch
import torch
from   torch import nn
from   torch.nn.modules.loss import _Loss
import torch.optim
from   torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from   pytorch_lightning.core.module import LightningModule

import csv
from   pathlib import Path
from   typing import Optional


class LitModel(LightningModule):
    def __init__(self,
    net: nn.Module,
    n_inputs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    loss: _Loss = nn.L1Loss(),
    lr_scheduling: Optional[bool] = False,
    stepsize: int = 20,
    gamma: float = 0.7848,
    metrics_dir: Optional[str] = None,
    track_image_files: bool = False
    ) -> None:
        super().__init__()
        
        self.net                = net
        self.n_inputs           = n_inputs
        self.lr                 = learning_rate
        self.wd                 = weight_decay
        self.loss               = loss
        self.err_metric         = nn.L1Loss()
        self.lr_scheduling      = lr_scheduling
        self.stepsize           = stepsize
        self.gamma              = gamma
        self.metrics_dir        = metrics_dir
        self.track_image_files  = track_image_files
        self.validation_results = []
        self.test_results       = []
        self.image_files        = []

        self.initialize()
        self.save_hyperparameters(ignore=['loss', 'net'])

    def initialize(self) -> None:
        # Create <metrics_dir/lr.csv> to log learning rate
        if self.lr_scheduling:

            # Check if <metrics_dir> exists
            if isinstance(self.metrics_dir, str) and Path(self.metrics_dir).is_dir():
                self.lr_csv = self.metrics_dir + '/lr.csv'
            else:
                raise ValueError('Invalid <metrics_dir>')

            # Check if <metrics_dir/lr.csv> already exsits, if yes: use latest learning rate
            if Path(self.lr_csv).is_file():
                logged_lrs = []

                with open(self.lr_csv, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        logged_lrs.append(row)
                
                latest_lr = float(logged_lrs[-1][-1])
                print('Use latest learning rate: {:.6f}'.format(latest_lr))
                self.lr = latest_lr
            else:
                # Create and initialize learning rate log file
                with open(self.lr_csv, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['epoch','lr'])
           
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.n_inputs == 1:
            return self.net(x)
        elif self.n_inputs == 2:
            return self.net(x, z)
        else:
            raise ValueError('Bad number of inputs.')
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        if self.n_inputs == 1:
            x, y   = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits  = self.forward(x, z)
        else:
            raise ValueError('Bad number of inputs.')
             
        # Calculate loss
        train_loss = self.loss(torch.squeeze(logits, dim=1), y)
        self.log('loss', train_loss)

        # Log learning rate
        if self.lr_scheduling and self.trainer.is_last_batch:
            
            # Log learning rate
            current_epoch = self.trainer.current_epoch
            current_lr    = self.trainer.optimizers[0].param_groups[0]['lr']
            
            with open(self.lr_csv, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([current_epoch,current_lr])

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        if self.n_inputs == 1:
            x, y   = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits  = self.forward(x, z)
        else:
            raise ValueError('Bad number of inputs.')
        
        y_true_np = torch.flatten(y).cpu().numpy()[0]
        y_pred_np = torch.flatten(logits)[0].cpu().numpy()
        self.validation_results.append([y_true_np, y_pred_np])
        
        # Calculate loss
        val_loss = self.loss(torch.squeeze(logits, dim=1), y)
        self.log('val_loss', val_loss)
        
        # Calculate error
        mae = self.err_metric(torch.squeeze(logits, dim=1), y)
        self.log('val_error', mae)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        if self.n_inputs == 1:
            if not self.track_image_files:
                x, y = batch
                logits = self.forward(x)

                y_true_np = torch.flatten(y)[0].cpu().numpy()
                y_pred_np = torch.flatten(logits)[0].cpu().numpy()
                self.test_results.append([y_true_np, y_pred_np])
            else:
                x, y, image_file = batch
                logits = self.forward(x)

                y_true_np = torch.flatten(y)[0].cpu().numpy()
                y_pred_np = torch.flatten(logits)[0].cpu().numpy()
                self.test_results.append([y_true_np, y_pred_np])
                self.image_files.append(image_file)
        elif self.n_inputs == 2:
            if not self.track_image_files:
                x, y, z = batch
                logits  = self.forward(x, z)

                y_true_np = torch.flatten(y)[0].cpu().numpy()
                y_pred_np = torch.flatten(logits)[0].cpu().numpy()
                z_np      = torch.flatten(z)[0].cpu().numpy()
                self.test_results.append([y_true_np, y_pred_np, z_np])
            else:
                x, y, z, image_file = batch
                logits  = self.forward(x, z)

                y_true_np = torch.flatten(y)[0].cpu().numpy()
                y_pred_np = torch.flatten(logits)[0].cpu().numpy()
                z_np      = torch.flatten(z)[0].cpu().numpy()
                self.test_results.append([y_true_np, y_pred_np, z_np])
                self.image_files.append(image_file)
        else:
            raise ValueError('Bad number of inputs.')
    
        # Calculate error
        mae = self.err_metric(torch.squeeze(logits, dim=1), y)
        self.log('test_error', mae)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # Automatic optimization
        if not self.lr_scheduling:
            return optimizer
        
        # Learning rate scheduling (manual optimization)
        elif self.lr_scheduling:
            scheduler = StepLR(optimizer, step_size=self.stepsize, gamma=self.gamma)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler}
                }
            """
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=0.000001)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                    'reduce_on_plateau': True,
                    'name': 'reduce_lr_on_plateau'
                    }
                }
            """
        else:
            raise ValueError('Bad value for argument "lr_scheduling". Got {}'.format(self.lr_scheduling))


class LitModel_Autoencoder(LightningModule):
    def __init__(self,
                 net,
                 n_inputs: int = 1,
                 lr: float = 3e-4) -> None:
        super().__init__()
        
        self.net      = net
        self.n_inputs = n_inputs
        self.lr       = lr
        self.loss     = nn.MSELoss()
        
        self.save_hyperparameters(ignore=['net'])
        
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if self.n_inputs == 1:
            return self.net(x)
        elif self.n_inputs == 2:
            return self.net(x, z)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        y      = torch.empty([])
        logits = torch.empty([])
        if self.n_inputs == 1:
            x, y   = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits  = self.forward(x, z)
        
        # Calculate loss
        loss = self.loss(logits, y)
        self.log('loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        y      = torch.empty([])
        logits = torch.empty([])
        if self.n_inputs == 1:
            x, y   = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits  = self.forward(x, z)
        else:
            raise ValueError('Bad number of inputs.')
        
        # Calculate loss
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)