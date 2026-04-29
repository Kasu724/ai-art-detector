"""Command-line entry points for project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_art_detector.config import load_experiment_config
from ai_art_detector.data.preparation import prepare_dataset
from ai_art_detector.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiad",
        description="AI art detector research project workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Scan raw images and build a reproducible dataset manifest.",
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to the top-level experiment config.",
    )
    prepare_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional override for the raw dataset directory.",
    )
    prepare_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional override for the output manifest CSV path.",
    )
    prepare_parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional override for the output summary JSON path.",
    )

    download_parser = subparsers.add_parser(
        "download-real-dataset",
        help="Download and materialize a supported real art-vs-AI dataset.",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/hf_art_images_ai_and_real"),
        help="Target directory for the downloaded images.",
    )
    download_parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional cap per dataset split for quick experiments.",
    )

    anime_download_parser = subparsers.add_parser(
        "download-anime-dataset",
        help="Download a style-focused anime human-vs-AI dataset mix for moderation experiments.",
    )
    anime_download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/anime_social_filter"),
        help="Target directory for the downloaded anime moderation dataset.",
    )
    anime_download_parser.add_argument(
        "--human-limit",
        type=int,
        default=3000,
        help="Total number of human-made anime images to materialize.",
    )
    anime_download_parser.add_argument(
        "--ai-limit",
        type=int,
        default=3000,
        help="Total number of AI-generated anime images to materialize.",
    )

    fanart_download_parser = subparsers.add_parser(
        "download-anime-fanart-dataset",
        help="Download a fanart-heavy anime moderation dataset with curated human examples.",
    )
    fanart_download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/anime_fanart_filter_v2"),
        help="Target directory for the curated fanart moderation dataset.",
    )
    fanart_download_parser.add_argument(
        "--human-limit",
        type=int,
        default=3000,
        help="Total number of human-made fanart images to materialize.",
    )
    fanart_download_parser.add_argument(
        "--ai-limit",
        type=int,
        default=3000,
        help="Total number of AI-generated anime images to materialize.",
    )

    fanart_v3_download_parser = subparsers.add_parser(
        "download-anime-fanart-v3-dataset",
        help="Download a fanart-heavy dataset with a broader AI-side source mix.",
    )
    fanart_v3_download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/anime_fanart_filter_v3"),
        help="Target directory for the broader fanart moderation dataset.",
    )
    fanart_v3_download_parser.add_argument(
        "--human-limit",
        type=int,
        default=3000,
        help="Total number of human-made fanart images to materialize.",
    )
    fanart_v3_download_parser.add_argument(
        "--ai-limit",
        type=int,
        default=3000,
        help="Total number of AI-generated anime images to materialize.",
    )

    fanart_v4_download_parser = subparsers.add_parser(
        "download-anime-fanart-v4-dataset",
        help="Download a stricter fanart dataset with auxiliary AI-generated Ghibli sources.",
    )
    fanart_v4_download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/anime_fanart_filter_v4"),
        help="Target directory for the AI-recall-focused fanart moderation dataset.",
    )
    fanart_v4_download_parser.add_argument(
        "--human-limit",
        type=int,
        default=4000,
        help="Total number of human-made fanart images to materialize.",
    )
    fanart_v4_download_parser.add_argument(
        "--ai-limit",
        type=int,
        default=4500,
        help="Total number of AI-generated anime images to materialize.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model from the prepared manifest.",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to the top-level experiment config.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on a manifest split.",
    )
    evaluate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to the top-level experiment config.",
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a `.pt` checkpoint created by the training pipeline.",
    )
    evaluate_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Manifest split to evaluate.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run single-image inference from a checkpoint or ONNX model.",
    )
    predict_parser.add_argument("--image", type=Path, required=True, help="Path to the image file.")
    predict_parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a PyTorch checkpoint.")
    predict_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    predict_parser.add_argument("--metrics-path", type=Path, default=None, help="Optional evaluation metrics JSON for threshold/calibration.")
    predict_parser.add_argument("--onnx-path", type=Path, default=None, help="Optional ONNX model path.")
    predict_parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override.")

    export_parser = subparsers.add_parser(
        "export-onnx",
        help="Export a PyTorch checkpoint to ONNX.",
    )
    export_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare multiple evaluation metrics files and write a summary artifact.",
    )
    compare_parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark-sample",
        help="Benchmark a held-out folder without adding it to the training manifest.",
    )
    benchmark_parser.add_argument("--sample-dir", type=Path, default=Path("sample"))
    benchmark_parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a PyTorch checkpoint.")
    benchmark_parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    benchmark_parser.add_argument("--metrics-path", type=Path, default=None, help="Optional evaluation metrics JSON.")
    benchmark_parser.add_argument("--onnx-path", type=Path, default=None, help="Optional ONNX model path.")
    benchmark_parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override.")

    return parser


def run_prepare_data(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)

    result = prepare_dataset(
        config=config,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest_path,
        summary_path=args.summary_path,
    )

    print(f"Prepared {result.num_records} records")
    print(f"Manifest: {result.manifest_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Run directory: {result.run_dir}")
    if result.invalid_files:
        print(f"Invalid files skipped: {len(result.invalid_files)}")
    return 0


def run_download_real_dataset(args: argparse.Namespace) -> int:
    from ai_art_detector.data.downloaders import download_real_art_dataset

    result = download_real_art_dataset(
        output_dir=args.output_dir,
        max_per_split=args.max_per_split,
    )
    print(f"Dataset source: {result['dataset_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Downloaded files: {result['num_downloaded']}")
    print(f"Summary path: {result['summary_path']}")
    return 0


def run_download_anime_dataset(args: argparse.Namespace) -> int:
    from ai_art_detector.data.downloaders import download_anime_social_dataset

    result = download_anime_social_dataset(
        output_dir=args.output_dir,
        human_limit=args.human_limit,
        ai_limit=args.ai_limit,
    )
    print(f"Dataset family: {result['dataset_family']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Downloaded files: {result['num_downloaded']}")
    print(f"Human images: {result['label_counts']['human']}")
    print(f"AI images: {result['label_counts']['ai']}")
    print(f"Summary path: {result['summary_path']}")
    return 0


def run_download_anime_fanart_dataset(args: argparse.Namespace) -> int:
    from ai_art_detector.data.downloaders import download_anime_fanart_dataset

    result = download_anime_fanart_dataset(
        output_dir=args.output_dir,
        human_limit=args.human_limit,
        ai_limit=args.ai_limit,
    )
    print(f"Dataset family: {result['dataset_family']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Downloaded files: {result['num_downloaded']}")
    print(f"Human images: {result['label_counts']['human']}")
    print(f"AI images: {result['label_counts']['ai']}")
    print(f"Summary path: {result['summary_path']}")
    return 0


def run_download_anime_fanart_v3_dataset(args: argparse.Namespace) -> int:
    from ai_art_detector.data.downloaders import download_anime_fanart_v3_dataset

    result = download_anime_fanart_v3_dataset(
        output_dir=args.output_dir,
        human_limit=args.human_limit,
        ai_limit=args.ai_limit,
    )
    print(f"Dataset family: {result['dataset_family']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Downloaded files: {result['num_downloaded']}")
    print(f"Human images: {result['label_counts']['human']}")
    print(f"AI images: {result['label_counts']['ai']}")
    print(f"Summary path: {result['summary_path']}")
    return 0


def run_download_anime_fanart_v4_dataset(args: argparse.Namespace) -> int:
    from ai_art_detector.data.downloaders import download_anime_fanart_v4_dataset

    result = download_anime_fanart_v4_dataset(
        output_dir=args.output_dir,
        human_limit=args.human_limit,
        ai_limit=args.ai_limit,
    )
    print(f"Dataset family: {result['dataset_family']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Downloaded files: {result['num_downloaded']}")
    print(f"Human images: {result['label_counts']['human']}")
    print(f"AI images: {result['label_counts']['ai']}")
    print(f"Summary path: {result['summary_path']}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    from ai_art_detector.training.pipeline import train_model

    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)
    result = train_model(config)
    print(f"Training run directory: {result.run_dir}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Final checkpoint: {result.final_checkpoint}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    from ai_art_detector.evaluation.pipeline import evaluate_checkpoint

    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)
    result = evaluate_checkpoint(config=config, checkpoint_path=args.checkpoint, split=args.split)
    print(f"Evaluation run directory: {result.run_dir}")
    print(f"Metrics: {result.metrics_path}")
    if result.predictions_path:
        print(f"Predictions: {result.predictions_path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    from ai_art_detector.inference.predictor import load_predictor

    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)
    predictor = load_predictor(
        config=config,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_path,
        metrics_path=args.metrics_path,
        threshold=args.threshold,
    )
    result = predictor.predict_file(args.image)
    print(result.to_json())
    return 0


def run_export_onnx(args: argparse.Namespace) -> int:
    from ai_art_detector.inference.onnx import export_checkpoint_to_onnx

    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)
    output_path = export_checkpoint_to_onnx(
        config=config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    print(f"Exported ONNX model: {output_path}")
    return 0


def run_compare_runs(args: argparse.Namespace) -> int:
    from ai_art_detector.evaluation.comparison import compare_evaluation_runs

    output_path = compare_evaluation_runs(args.metrics, args.output)
    print(f"Comparison summary: {output_path}")
    print(f"Comparison markdown: {Path(output_path).with_suffix('.md')}")
    return 0


def run_benchmark_sample(args: argparse.Namespace) -> int:
    from ai_art_detector.evaluation.sample_benchmark import benchmark_sample_folder

    config = load_experiment_config(args.config)
    configure_logging(config.runtime.log_level)
    result = benchmark_sample_folder(
        config=config,
        sample_dir=args.sample_dir,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics_path,
        onnx_path=args.onnx_path,
        threshold=args.threshold,
    )
    print(f"Benchmark run directory: {result.run_dir}")
    print(f"Accuracy: {result.correct}/{result.total} ({result.accuracy:.4f})")
    print(f"Summary: {result.summary_path}")
    print(f"Predictions: {result.predictions_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        return run_prepare_data(args)
    if args.command == "download-real-dataset":
        return run_download_real_dataset(args)
    if args.command == "download-anime-dataset":
        return run_download_anime_dataset(args)
    if args.command == "download-anime-fanart-dataset":
        return run_download_anime_fanart_dataset(args)
    if args.command == "download-anime-fanart-v3-dataset":
        return run_download_anime_fanart_v3_dataset(args)
    if args.command == "download-anime-fanart-v4-dataset":
        return run_download_anime_fanart_v4_dataset(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "predict":
        return run_predict(args)
    if args.command == "export-onnx":
        return run_export_onnx(args)
    if args.command == "compare-runs":
        return run_compare_runs(args)
    if args.command == "benchmark-sample":
        return run_benchmark_sample(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
