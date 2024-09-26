import torch
import pandas as pd
import os
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, data_dir, beat_annotations, tempo_annotation, transform=None):
        self.data_dir = data_dir
        self.beat_annotations = beat_annotations
        self.tempo_annotation = tempo_annotation
        self.transform = transform
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.beat = [csv for csv in os.listdir(beat_annotations) if csv.endswith('.beats')]
        self.tempo = [csv for csv in os.listdir(tempo_annotation) if csv.endswith('.bpm')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name_audio = os.path.join(self.data_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(file_name_audio, normalize=True)  # scale data to [-1, 1]

        # spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=220, normalized=True)(waveform)
        #
        # spectrogram = spectrogram.unsqueeze(1)
        # spectrogram = torch.nn.functional.interpolate(spectrogram, size=(81, 3000), mode='bilinear')
        #
        # fbanks = torchaudio.functional.linear_fbanks(n_freqs=81, f_min=30.0, f_max=(sample_rate // 2),
        #                                              sample_rate=sample_rate, n_filter=81)
        #
        # spectrogram = torch.matmul(fbanks, spectrogram)  # application du filtre
        #
        # spec_log = torchaudio.transforms.AmplitudeToDB(stype="amplitude")(spectrogram)  # passage à l'échelle log
        # spec_log = (spec_log - spec_log.mean()) / spec_log.std()  # z-score normalisation : mean = 0, std =1
        #
        # spec_log = spec_log.transpose(2, 3)
        #
        # n_frames = spec_log.shape[-2]

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=220, f_min=30,
                                                        f_max=(sample_rate // 2), n_mels=81, normalized=True)(waveform)

        mel_spec = mel_spec.unsqueeze(1)
        mel_spec = torch.nn.functional.interpolate(mel_spec, size=(81, 3000), mode='bilinear')
        mel_spec = mel_spec.transpose(2, 3)

        n_frames = mel_spec.shape[-2]

        file_name_beat = os.path.join(self.beat_annotations, self.beat[idx])
        csv_beat = pd.read_csv(file_name_beat, sep='\t', header=None)
        beat_frames = (csv_beat[0] * 100).astype(int).tolist()  # à partir de la seconde ou l'on a un beat on recupère le numéro de la frame correspodante
        target_beat = torch.zeros(n_frames)

        file_name_tempo = os.path.join(self.tempo_annotation, self.tempo[idx])
        csv_tempo = pd.read_csv(file_name_tempo, sep='e', header=None)
        tempo_value = round((csv_tempo[0]*pow(10, csv_tempo[1]))).astype(int).item()
        target_tempo = torch.zeros(300)

        if tempo_value >= 300:
            tempo_value = 299

        target_tempo[tempo_value] = 1

        # try:
        #     target_tempo[tempo_value - 1] = 0.5
        # except IndexError:
        #     pass
        # try:
        #     target_tempo[tempo_value + 1] = 0.5
        # except IndexError:
        #     pass
        # try:
        #     target_tempo[tempo_value - 2] = 0.25
        # except IndexError:
        #     pass
        # try:
        #     target_tempo[tempo_value + 2] = 0.25
        # except IndexError:
        #     pass

        for i in beat_frames:  # on remplace dans le vecteur les frames correspondantes par la valeur 1 qui correspond à la présence d'un beat
            if i >= n_frames:  # if beat annotations "overflow" the length size of resized spectrogram
                pass
            else:
                target_beat[i] = 1

        # return spec_log, target_beat, target_tempo
        return mel_spec, target_beat, target_tempo
