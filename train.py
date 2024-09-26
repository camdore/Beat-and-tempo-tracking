from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from torchmetrics.classification import F1Score, Accuracy, BinaryAccuracy, BinaryF1Score
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import JointModel
from pipeline_utils import AudioDataset
from madmom.features import DBNBeatTrackingProcessor
from utils import BeatF1Score
from misc import check_all_distrib, check_input_distribution


class DataModule(L.LightningDataModule):  # ce module s'occupe de toute la mise en forme de données et retourne les différents dataloaders
    def __init__(self, path_track, path_beat, path_tempo, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.path_track = path_track
        self.path_beat = path_beat
        self.path_tempo = path_tempo
        self.batch_size = batch_size

        self.data = AudioDataset(data_dir=self.path_track, beat_annotations=self.path_beat, tempo_annotation=self.path_tempo)

        self.train, self.test = random_split(self.data, lengths=[0.75, 0.25])  # repartition avec des ratios classiques pour les différents datasets

        self.train, self.val = random_split(self.train, lengths=[0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=4, shuffle=False, persistent_workers=True)


class LitModel(L.LightningModule):  # ce module s'occupe de la définition de toutes les étapes de l'entrainement de manière centralisée
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = JointModel()
        self.loss_fn = nn.BCEWithLogitsLoss()  # la loss est binaire car la sortie de beat est un sigmoid
        self.loss_fn2 = nn.CrossEntropyLoss()  # on choisi cette loss pour la sortie tempo qui est un softmax
        self.f1_metric = BeatF1Score(window_size=70)  # custom f1 score pour le beat
        self.accuracy = Accuracy(task='multiclass', num_classes=300)  # accuracy pour le tempo
        self.automatic_optimization = True
        self.post_processor = DBNBeatTrackingProcessor(min_bpm=55, max_bpm=215, fps=100, online=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def training_step(self, train_batch, batch_idx):
        x, y_beat, y_tempo = train_batch
        x = x.squeeze(1)
        pred_beat, pred_tempo = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('train_lossBCE_beat', loss_beat, on_step=True, on_epoch=False)

        loss_tempo = self.loss_fn2(pred_tempo, y_tempo)
        self.log('train_lossCE_tempo', loss_tempo, on_step=True, on_epoch=False)

        final_loss = (loss_beat + loss_tempo) / 2
        self.log('train_final_loss', final_loss, on_step=True, on_epoch=False)

        return loss_beat

    def validation_step(self, validate_batch, batch_idx):
        x, y_beat, y_tempo = validate_batch
        x = x.squeeze(1)
        pred_beat, pred_tempo = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('val_lossBCE_beat', loss_beat, on_step=False, on_epoch=True)

        loss_tempo = self.loss_fn2(pred_tempo, y_tempo)
        self.log('val_lossCE_tempo', loss_tempo, on_step=False, on_epoch=True)

        final_loss = (loss_beat + loss_tempo) / 2
        self.log('val_final_loss', final_loss, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x, y_beat, y_tempo = test_batch
        x = x.squeeze(1)
        pred_beat, pred_tempo = self.model(x)

        loss_beat = self.loss_fn(pred_beat, y_beat)
        self.log('test_lossBCE_beat', loss_beat, on_step=False, on_epoch=True)

        pred_beat_og = torch.sigmoid(pred_beat)

        # POST-PROCESSING
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

        # Update the custom F1 metric
        self.f1_metric.update(pred_beat_tensor, y_beat)
        f1_score = self.f1_metric.compute()
        self.log('test_f1score_beat', f1_score, on_step=True, on_epoch=True)

        self.accuracy(pred_tempo, y_tempo)
        self.log('test_accuracy_tempo', self.accuracy, on_step=True, on_epoch=True)

        loss_tempo = self.loss_fn2(pred_tempo, y_tempo)
        self.log('test_lossCE_tempo', loss_tempo, on_step=False, on_epoch=True)

        final_loss = (loss_beat + loss_tempo) / 2
        self.log('test_final_loss', final_loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        # Reset the metric for the next epoch
        self.f1_metric.reset()
        self.accuracy.reset()


def run_baseline(batch_size, path_track, path_beat, path_tempo):
    logger = TensorBoardLogger(save_dir='.', default_hp_metric=True, log_graph=True, version='updated_model')

    L.seed_everything(42, workers=True)

    # checkpoint_path = 'C:/Users/camil/Desktop/MWM/test MWM/lightning_logs/updated_model/checkpoints/epoch=149-step=1500.ckpt'

    # model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model = LitModel()
    # print(model.eval())

    data = DataModule(path_track, path_beat, path_tempo, batch_size)

    # check_all_distrib(data)

    trainer = L.Trainer(
        accelerator='cuda',
        max_epochs=150,
        log_every_n_steps=10,
        logger=False,
        precision='16-mixed',
        gradient_clip_val=0.5,
        gradient_clip_algorithm='norm',
        callbacks=[EarlyStopping(monitor='val_lossBCE_beat', patience=50, mode='min'),
                   # ModelCheckpoint(monitor='val_lossBCE_beat', mode='min', save_top_k=3),
                   ],
        deterministic=True
    )

    # on appelle chaque step définies dans LitModel, Lightning s'occupe du reste avec notamment la remise à zéro
    # des gradients, le backward pass ainsi que l'allocation des données sur le bon device.
    trainer.fit(model, data)
    trainer.validate(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    path_track = 'C:/Users/camil/Desktop/Thales/GTZAN/tracks'
    path_tempo = 'C:/Users/camil/Desktop/Thales/GTZAN/tempo'
    path_beat = 'C:/Users/camil/Desktop/Thales/GTZAN/beats'
    batch_size = 64

    parser = ArgumentParser()

    parser.add_argument('--batch-size', default=batch_size, type=int)
    parser.add_argument('--path-track', default=path_track, type=str)
    parser.add_argument('--path-beat', default=path_beat, type=str)
    parser.add_argument('--path-tempo', default=path_tempo, type=str)

    args = parser.parse_args()

    run_baseline(args.batch_size, args.path_track, args.path_beat, args.path_tempo)
