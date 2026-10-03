from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sleep_monitor.evaluation.parser import VideoAnnotation
from sleep_monitor.schemas.report import FinalReport
from sleep_monitor.schemas.state import ActivityState


class Evaluator:
    """Computes evaluation metrics between system output and ground truth."""

    def __init__(
        self,
        annotation: VideoAnnotation,
        report: FinalReport,
        tolerance_sec: float = 10.0,
    ):
        self.annotation = annotation
        self.report = report
        self.tolerance_sec = tolerance_sec
        self.all_states = [s.value for s in ActivityState] + ["out_of_bed"]

    def _sample_timeline(
        self, segments, duration_sec: float, sample_rate_hz: float = 1.0
    ) -> list[str]:
        """Convert a list of segments into a fixed-rate sequence of labels."""
        num_samples = int(duration_sec * sample_rate_hz)
        timeline = ["unknown"] * num_samples

        for seg in segments:
            # Handle both ground truth (dict/BaseModel) and predicted (TimelineSegment)
            start = getattr(seg, "start_time", getattr(seg, "start_time_sec", 0))
            end = getattr(seg, "end_time", getattr(seg, "end_time_sec", 0))
            state = getattr(seg, "state", "unknown").lower()

            start_idx = min(int(start * sample_rate_hz), num_samples - 1)
            end_idx = min(int(end * sample_rate_hz), num_samples)

            for i in range(start_idx, end_idx):
                timeline[i] = state

        return timeline

    def evaluate_activity_classification(self) -> dict[str, Any]:
        """Compute frame-level accuracy, precision, recall, and confusion matrix."""
        duration = self.report.observation_duration_sec
        if duration <= 0:
            duration = max([s.end_time for s in self.annotation.segments] + [0])

        y_true = self._sample_timeline(self.annotation.segments, duration)
        y_pred = self._sample_timeline(self.report.timeline, duration)

        # Ensure they are the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        labels = sorted(set(y_true + y_pred))

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
            "labels": labels,
        }
        return metrics

    def evaluate_bed_events(self, event_type: str) -> dict[str, Any]:
        """Compute precision and recall for bed events with a temporal tolerance."""
        true_events = [
            e.time
            for e in self.annotation.events
            if e.type.lower() == event_type.lower()
        ]
        pred_events = [
            e.confirmed_at_sec
            for e in self.report.bed_events
            if e.event_type.lower() == event_type.lower()
        ]

        true_events.sort()
        pred_events.sort()

        matched_true = set()
        matched_pred = set()

        for i, p_time in enumerate(pred_events):
            for j, t_time in enumerate(true_events):
                if j in matched_true:
                    continue
                if abs(p_time - t_time) <= self.tolerance_sec:
                    matched_true.add(j)
                    matched_pred.add(i)
                    break

        tp = len(matched_true)
        fp = len(pred_events) - tp
        fn = len(true_events) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
        }

    def evaluate_durations(self) -> pd.DataFrame:
        """Compare predicted vs ground truth duration per activity."""
        duration = self.report.observation_duration_sec
        if duration <= 0:
            duration = max([s.end_time for s in self.annotation.segments] + [0])

        y_true = self._sample_timeline(self.annotation.segments, duration)
        y_pred = self._sample_timeline(self.report.timeline, duration)

        from collections import Counter

        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)

        all_labels = set(true_counts.keys()).union(set(pred_counts.keys()))

        data = []
        for label in all_labels:
            # 1 sample = 1 second (default sample rate)
            gt_dur = true_counts.get(label, 0)
            pred_dur = pred_counts.get(label, 0)
            error = abs(pred_dur - gt_dur)

            data.append(
                {
                    "Activity": label,
                    "Ground Truth (sec)": gt_dur,
                    "Predicted (sec)": pred_dur,
                    "Absolute Error (sec)": error,
                }
            )

        return pd.DataFrame(data).sort_values("Activity")

    def get_full_evaluation(self) -> dict[str, Any]:
        """Run all evaluations and return a consolidated dictionary."""
        durations_df = self.evaluate_durations()
        return {
            "classification": self.evaluate_activity_classification(),
            "events": {
                "bed_exit": self.evaluate_bed_events("bed_exit"),
                "return_to_bed": self.evaluate_bed_events("return_to_bed"),
            },
            "durations": durations_df.to_dict(orient="records"),
        }
