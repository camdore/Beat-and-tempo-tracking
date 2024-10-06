from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from model import TCNModel
from pipeline_utils import AudioDataset
from madmom.features import DBNBeatTrackingProcessor
from metrics import BeatF1Score


class DataModule(L.LightningDataModule):  # ce module s'occupe de toute la mise en forme de données et retourne les différents dataloaders
    def __init__(self, path_track, path_beat, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.path_track = path_track
        self.path_beat = path_beat
        self.batch_size = batch_size

        self.data = AudioDataset(data_dir=self.path_track, beat_annotations=self.path_beat)

        self.train, self.test = random_split(self.data, lengths=[0.75, 0.25])  # repartition avec des ratios classiques pour les différents datasets

        self.train, self.val = random_split(self.train, lengths=[0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, shuffle=False, persistent_workers=True)


class LitModel(L.LightningModule):  # ce module s'occupe de la définition de toutes les étapes de l'entrainement de manière centralisée
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TCNModel()
        self.loss_fn = nn.BCEWithLogitsLoss()  # la loss est binaire car la sortie de beat est un sigmoid
        self.f1_metric = BeatF1Score(window_size=70)  # custom f1 score pour le beat
        self.automatic_optimization = True
        self.post_processor = DBNBeatTrackingProcessor(min_bpm=55, max_bpm=215, fps=100, online=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_plateau,
                "interval": "epoch",
                "monitor": "val_lossBCE",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                "name": "lr_scheduler_lossBCE",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y_beat = train_batch
        x = x.squeeze(1)
        pred_beat = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('train_lossBCE', loss_beat, on_step=True, on_epoch=False)

        return loss_beat

    def validation_step(self, validate_batch, batch_idx):
        x, y_beat = validate_batch
        x = x.squeeze(1)
        pred_beat = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('val_lossBCE', loss_beat, on_step=False, on_epoch=True)


def run_baseline(batch_size, path_track, path_beat):
    logger = TensorBoardLogger(save_dir='.', default_hp_metric=True, log_graph=True, version='beat_tracking')

    L.seed_everything(42, workers=True)

    model = LitModel()

    data = DataModule(path_track, path_beat, batch_size)

    trainer = L.Trainer(
        accelerator='cuda',
        max_epochs=150,
        log_every_n_steps=10,
        logger=logger,
        precision='16-mixed',
        gradient_clip_val=0.5,
        gradient_clip_algorithm='norm',
        callbacks=[EarlyStopping(monitor='val_lossBCE', patience=50, mode='min'),
                   LearningRateMonitor(logging_interval='epoch'),
                   ],
        deterministic=True
    )

    # on appelle chaque step définies dans LitModel, Lightning s'occupe du reste avec notamment la remise à zéro
    # des gradients, le backward pass ainsi que l'allocation des données sur le bon device.
    trainer.fit(model, data)
    trainer.validate(model, data)


if __name__ == "__main__":
    path_track = 'path/to/GTZAN/tracks'
    path_beat = 'path/to/GTZAN/beats'
    batch_size = 64

    parser = ArgumentParser()

    parser.add_argument('--batch-size', default=batch_size, type=int)
    parser.add_argument('--path-track', default=path_track, type=str)
    parser.add_argument('--path-beat', default=path_beat, type=str)

    args = parser.parse_args()

    run_baseline(args.batch_size, args.path_track, args.path_beat)
