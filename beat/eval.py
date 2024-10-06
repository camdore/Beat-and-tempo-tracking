from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch
from argparse import ArgumentParser

from model import TCNModel
from pipeline_utils import AudioDataset
from madmom.features import DBNBeatTrackingProcessor
from metrics import BeatF1Score


class DataModule(L.LightningDataModule):  # ce module s'occupe de toute la mise en forme de données et retourne les différents dataloaders
    def __init__(self, path_track, path_beat):
        super().__init__()
        self.save_hyperparameters()
        self.path_track = path_track
        self.path_beat = path_beat
 
        self.data = AudioDataset(data_dir=self.path_track, beat_annotations=self.path_beat)

        self.train, self.test = random_split(self.data, lengths=[0.75, 0.25])  # repartition avec des ratios classiques pour les différents datasets

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=4, shuffle=False, persistent_workers=True)


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

    def test_step(self, test_batch, batch_idx):
        x, y_beat = test_batch
        x = x.squeeze(1)
        pred_beat = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('test_lossBCE', loss_beat, on_step=False, on_epoch=True)

        pred_beat_og = torch.sigmoid(pred_beat)

        # POST-PROCESSING BEAT --------------------------------------------------------------
        pred_beat = pred_beat_og.to(device='cpu')
        pred_beat = pred_beat.numpy()
        pred_beat = pred_beat.squeeze(0)
        self.post_processor.reset()
        pred_beat_processed = self.post_processor.process_online(pred_beat)
        pred_beat_processed = (pred_beat_processed * 100).astype(int).tolist()

        pred_beat_tensor = torch.zeros(x.shape[-2])
        for i in pred_beat_processed:
            pred_beat_tensor[i] = 1

        pred_beat_tensor = pred_beat_tensor.unsqueeze(0)
        pred_beat_tensor = pred_beat_tensor.to(device='cuda')

        # BEAT METRIC -----------------------------------------------------------------------
        self.f1_metric.update(pred_beat_tensor, y_beat)
        f1_score = self.f1_metric.compute()
        self.log('test_f1score', f1_score, on_step=True, on_epoch=True)

    def on_test_epoch_end(self):
        # Reset the metric for the next epoch
        self.f1_metric.reset()



def run_baseline(path_track, path_beat, checkpoint_path):
    logger = TensorBoardLogger(save_dir='.', default_hp_metric=True, log_graph=True, version='beat_tracking_eval')

    L.seed_everything(42, workers=True)

    model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    data = DataModule(path_track, path_beat)

    trainer = L.Trainer(
        accelerator='cuda',
        log_every_n_steps=10,
        logger=logger,
        precision='16-mixed',
        deterministic=True
    )

    trainer.test(model, data)


if __name__ == "__main__":
    path_track = 'path/to/GTZAN/tracks'
    path_beat = 'path/to/GTZAN/beats'
    checkpoint_path = 'path/to/Beat-and-tempo-tracking//checkpoints/TCN_beat_only.ckpt'

    parser = ArgumentParser()

    parser.add_argument('--path-track', default=path_track, type=str)
    parser.add_argument('--path-beat', default=path_beat, type=str)
    parser.add_argument('--path-checkpoint', default=checkpoint_path, type=str)

    args = parser.parse_args()

    run_baseline(args.path_track, args.path_beat, args.path_checkpoint)
