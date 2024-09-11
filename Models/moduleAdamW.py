import pytorch_lightning as pl
from torch import nn
from torch import optim

class AdamWModule(pl.LightningModule):
    def __init__(self, n_features: int, model: nn.Module, criterion, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.model = model(n_features)
        self.criterion = criterion
        self.lr = learning_rate

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())