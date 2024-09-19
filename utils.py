import torch
import torchmetrics


class BeatF1Score(torchmetrics.Metric):
    def __init__(self, window_size: int = 70, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # The window size in frames (±70 frames around the annotation)
        self.window_size = window_size

        # State variables to track true positives, false positives, and false negatives
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the state with predictions and targets.
        preds: Predicted beat times (binary, shape: [batch_size, seq_length])
        target: Ground truth beat times (binary, shape: [batch_size, seq_length])
        """
        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred_beats = torch.nonzero(preds[i]).squeeze()  # Indices of predicted beats (frame numbers)
            true_beats = torch.nonzero(target[i]).squeeze()  # Indices of true beats (frame numbers)

            if len(true_beats) == 0:
                # No true beats in this sample
                self.false_positives += len(pred_beats)
                continue

            if len(pred_beats) == 0:
                # No predicted beats, count all as false negatives
                self.false_negatives += len(true_beats)
                continue

            # Calculate true positives and false positives
            tp = 0
            used_true_beats = torch.zeros_like(true_beats)  # Track which true beats have been matched
            for pred_beat in pred_beats:
                # Find the closest true beat within the window of ±70 frames
                beat_diffs = torch.abs(true_beats - pred_beat)
                min_diff, min_idx = torch.min(beat_diffs, dim=0)

                if min_diff <= self.window_size and not used_true_beats[min_idx]:
                    tp += 1
                    used_true_beats[min_idx] = 1  # Mark this true beat as matched
                elif 0 < min_diff <= 2 and not used_true_beats[min_idx]:
                    tp += 0.5  # Adjacent match within ±2 frames gets weight of 0.5
                    used_true_beats[min_idx] = 1  # Mark this true beat as matched
                else:
                    self.false_positives += 1

            self.true_positives += tp
            self.false_negatives += len(true_beats) - tp  # Unmatched true beats are false negatives

    def compute(self):
        """
        Compute the F1 score based on the accumulated state variables.
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return f1_score

    def reset(self):
        """
        Reset the metric state.
        """
        self.true_positives = torch.tensor(0.0)
        self.false_positives = torch.tensor(0.0)
        self.false_negatives = torch.tensor(0.0)
