import pandas as pd
import pytest

from sleep_monitor.evaluation.metrics import Evaluator
from sleep_monitor.evaluation.parser import (
    GroundTruthEvent,
    GroundTruthSegment,
    VideoAnnotation,
)
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.report import (
    BedSummary,
    FinalReport,
    TargetPersonInfo,
    TimelineSegment,
)


@pytest.fixture
def mock_annotation():
    return VideoAnnotation(
        video_file="test.mp4",
        annotator="test",
        segments=[
            GroundTruthSegment(start_time=0.0, end_time=10.0, state="lying_in_bed"),
            GroundTruthSegment(start_time=10.0, end_time=20.0, state="sitting_on_bed"),
            GroundTruthSegment(start_time=20.0, end_time=30.0, state="walking"),
        ],
        events=[GroundTruthEvent(type="bed_exit", time=22.0)],
    )


@pytest.fixture
def mock_report():
    return FinalReport(
        observation_duration_sec=30.0,
        target_person=TargetPersonInfo(final_state="walking"),
        activity_durations={
            "lying_in_bed": 10.0,
            "sitting_on_bed": 12.0,
            "walking": 8.0,
        },
        bed_summary=BedSummary(
            time_in_bed=22.0,
            time_out_of_bed=8.0,
            exit_count=1,
            return_count=0,
            longest_out_of_bed=8.0,
        ),
        bed_events=[
            BedEvent(
                event_type="BED_EXIT",
                timestamp_sec=20.0,
                confirmed_at_sec=25.0,
                previous_state="sitting_on_bed",
                current_state="walking",
                confidence=0.9,
                decision="MONITOR",
            )
        ],
        timeline=[
            TimelineSegment(
                start_time_sec=0.0, end_time_sec=10.0, state="lying_in_bed"
            ),
            TimelineSegment(
                start_time_sec=10.0, end_time_sec=22.0, state="sitting_on_bed"
            ),
            TimelineSegment(start_time_sec=22.0, end_time_sec=30.0, state="walking"),
        ],
        safety_decision="MONITOR",
    )


def test_activity_classification(mock_annotation, mock_report):
    evaluator = Evaluator(mock_annotation, mock_report)
    metrics = evaluator.evaluate_activity_classification()

    assert "accuracy" in metrics
    assert "report" in metrics
    assert "confusion_matrix" in metrics

    # 0-10: lying vs lying (10s correct)
    # 10-20: sitting vs sitting (10s correct)
    # 20-22: walking vs sitting (2s incorrect)
    # 22-30: walking vs walking (8s correct)
    # Total 28s correct out of 30s = 0.9333...
    assert metrics["accuracy"] > 0.9


def test_bed_events(mock_annotation, mock_report):
    evaluator = Evaluator(mock_annotation, mock_report, tolerance_sec=5.0)
    # Ground truth event time is 22.0
    # Predicted event confirmed_at_sec is 25.0
    # Difference is 3.0 seconds <= 5.0, so it should be a match

    metrics = evaluator.evaluate_bed_events("bed_exit")
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_durations(mock_annotation, mock_report):
    evaluator = Evaluator(mock_annotation, mock_report)
    df = evaluator.evaluate_durations()

    assert isinstance(df, pd.DataFrame)

    # Sitting ground truth: 10, predicted: 12. Error: 2
    sitting_row = df[df["Activity"] == "sitting_on_bed"].iloc[0]
    assert sitting_row["Absolute Error (sec)"] == 2.0
