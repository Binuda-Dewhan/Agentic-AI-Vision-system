import json
import logging
from pathlib import Path

import click

from sleep_monitor.config.settings import Settings
from sleep_monitor.utils.logging import setup_logging


@click.group()
def cli():
    """Elderly Sleep Monitoring System CLI."""


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file.",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None, help="Output JSON file path."
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging."
)
@click.option(
    "--visualize", is_flag=True, default=False, help="Show video frames during analysis."
)
def analyze(video_path, config, output, verbose, visualize):
    """Analyze a video file for elderly sleep monitoring."""
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

    # Load configuration
    if config:
        settings = Settings.from_yaml(config)
        logger.info(f"Loaded config from {config}")
    else:
        settings = Settings()
        logger.info("Using default configuration")

    # Set default output path if not provided
    if output is None:
        video_name = Path(video_path).stem
        output = f"output/{video_name}_report.json"

    # Run the pipeline
    from sleep_monitor.pipeline import Pipeline

    pipeline = Pipeline(settings)
    report = pipeline.run(video_path, output_path=output, visualize=visualize)

    # Print summary to console
    click.echo("\n" + "=" * 60)
    click.echo("ANALYSIS COMPLETE")
    click.echo("=" * 60)
    click.echo(f"Video duration: {report.get('observation_duration_sec', 0):.1f}s")
    click.echo(f"Safety decision: {report.get('safety_decision', 'N/A')}")
    click.echo(f"Timeline segments: {len(report.get('timeline', []))}")
    click.echo(f"Bed events: {len(report.get('bed_events', []))}")

    if report.get("activity_durations"):
        click.echo("\nActivity Durations:")
        for state, dur in report["activity_durations"].items():
            click.echo(f"  {state}: {dur:.1f}s")

    if report.get("bed_summary"):
        bs = report["bed_summary"]
        click.echo("\nBed Summary:")
        click.echo(f"  Time in bed: {bs.get('time_in_bed', 0):.1f}s")
        click.echo(f"  Time out of bed: {bs.get('time_out_of_bed', 0):.1f}s")
        click.echo(f"  Exit count: {bs.get('exit_count', 0)}")
        click.echo(f"  Return count: {bs.get('return_count', 0)}")
        click.echo(f"  Longest out: {bs.get('longest_out_of_bed', 0):.1f}s")

    click.echo(f"\nFull report saved to: {output}")


@cli.command()
@click.argument("report_path", type=click.Path(exists=True))
@click.argument("annotation_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="output/eval",
    help="Output directory for evaluation results.",
)
def evaluate(report_path, annotation_path, output_dir):
    """Evaluate pipeline report against ground truth annotations."""
    from sleep_monitor.evaluation.metrics import Evaluator
    from sleep_monitor.evaluation.parser import load_annotations
    from sleep_monitor.evaluation.visualizer import (
        plot_confusion_matrix,
        plot_timeline_comparison,
    )
    from sleep_monitor.schemas.report import FinalReport

    click.echo(f"Evaluating {report_path} against {annotation_path}...")

    # Load data
    annotations = load_annotations(annotation_path)
    with open(report_path, "r") as f:
        report_data = json.load(f)
    report = FinalReport(**report_data)

    # Evaluate
    evaluator = Evaluator(annotations, report)
    results = evaluator.get_full_evaluation()

    # Output metrics
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Visualizations
    plot_confusion_matrix(results, str(out_dir / "confusion_matrix.png"))

    # Timeline
    duration = report.observation_duration_sec
    y_true = evaluator._sample_timeline(annotations.segments, duration)
    y_pred = evaluator._sample_timeline(report.timeline, duration)

    min_len = min(len(y_true), len(y_pred))
    plot_timeline_comparison(
        results, y_true[:min_len], y_pred[:min_len], str(out_dir / "timeline.png")
    )

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION COMPLETE")
    click.echo("=" * 60)
    click.echo(f"Accuracy: {results['classification']['accuracy']:.2%}")
    click.echo(f"Bed Exit Recall: {results['events']['bed_exit']['recall']:.2%}")
    click.echo(f"Bed Exit Precision: {results['events']['bed_exit']['precision']:.2%}")
    click.echo(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    cli()
