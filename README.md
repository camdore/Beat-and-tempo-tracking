# Beat and tempo tracking


### Beat Tracking (Working)
In the folder beat is the implementation of the paper : [Temporal convolutional networks for musical audio beat tracking](https://www.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570533824.pdf) by Matthew E. P. Davies and Sebastian Böck.

### Beat and tempo tracking (in progress / stuck)

In the folder joint_beat_tempo tries to follow the paper [Multi-task learning of tempo and beat learning one the improve the other](https://archives.ismir.net/ismir2019/paper/000058.pdf) by Sebastian Böck, Matthew E. P. Davies and Peter Knees.

## User guide 

You can clone this repository on your machine with the following line :  

    git clone git@github.com:camdore/Beat-and-tempo-tracking.git

In order to this repository to work you need several packages that you can install with the following command :

    pip install -r requirements.txt

## Training

To train the model described in the paper, run this command :

    python train.py --batch-size 64 --path-track "path/to/tracks" --path-beats "path/to/beats" 

The hyperparameters are already in the code (learning rate, window size of F1 score)

These are for the beat only version, for the joint beat and tempo tracking add the argument --path-tempo.

## Evaluation

To evaluate my model on a dataset, run :

    python eval.py --path-track "path/to/tracks" --path-beats "path/to/beats --checkpoint-path "path/to/the/checkpoints/TCN_beat_only.ckpt"

No batch size because parameter post processing only accept batch size of 1.  
These are for the beat only version, for the joint beat and tempo tracking add the argument --path-tempo adn change the checkpoint-path parameter.

## Pre-trained Models

The pretrained models weight are in the folder of this repository named checkpoints.

## Results

Our model achieves the following performance on : 

### GTZAN

| Model name         | F1-score  |
| ------------------ |---------- |
| TCNModel (ours)    |  0.823    |
| PaperModel         |  0.843    |
