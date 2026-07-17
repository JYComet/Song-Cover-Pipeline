#!/usr/bin/env python3
"""
Song Cover Pipeline — Main Entry Point

Usage:
    python run_pipeline.py [--config CONFIG_YAML] [--force]

Examples:
    # Run with default config
    python run_pipeline.py

    # Run with a specific config
    python run_pipeline.py --config configs/my_task.yaml

    # Force re-run all stages (ignore checkpoint)
    python run_pipeline.py --force
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root and src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    import yaml

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Inject project root for path resolution
    config["_project_root"] = str(PROJECT_ROOT)

    logger.info("Config loaded: %s", config_path.name)
    return config


def validate_config(config: dict) -> list[str]:
    """
    Validate the configuration and return a list of warnings/issues.

    Does NOT stop execution — just reports potential problems.
    """
    warnings = []

    # All relative paths in the config are resolved against PROJECT_ROOT.
    def _resolve_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else PROJECT_ROOT / pp

    # Check harmony separation config
    hs = config.get("harmony_separation", {})
    if hs.get("enabled", True):
        ckpt = hs.get("checkpoint_path", "")
        if ckpt and not _resolve_path(ckpt).is_file():
            warnings.append(
                f"Harmony separation checkpoint not found: {ckpt}\n"
                f"  Download it from MSST model repository and place in the specified path."
            )
        cfg = hs.get("config_path", "")
        if cfg and not _resolve_path(cfg).is_file():
            warnings.append(f"Harmony separation config not found: {cfg}")

    # Check reverb separation config
    rs = config.get("reverb_separation", {})
    if rs.get("enabled", True):
        ckpt = rs.get("checkpoint_path", "")
        if ckpt and not _resolve_path(ckpt).is_file():
            warnings.append(f"Reverb separation checkpoint not found: {ckpt}")
        cfg = rs.get("config_path", "")
        if cfg and not _resolve_path(cfg).is_file():
            warnings.append(f"Reverb separation config not found: {cfg}")

    # Check timbre conversion config
    tc = config.get("timbre_conversion", {})
    if tc.get("enabled", True):
        ddsp_dir = tc.get("ddsp_project_dir", "")
        if ddsp_dir and not _resolve_path(ddsp_dir).is_dir():
            warnings.append(f"DDSP project directory not found: {ddsp_dir}")
        model = tc.get("model_ckpt", "")
        # Model checkpoint path is resolved against CWD (project root),
        # NOT against ddsp_project_dir (see DDSPConverter.__init__).
        if model:
            model_path = _resolve_path(model)
            if not model_path.is_file():
                warnings.append(f"DDSP model checkpoint not found: {model_path}")

    # Check input song
    input_song = config.get("task", {}).get("input_song", "")
    if input_song:
        input_path = Path(input_song)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        if not input_path.is_file():
            warnings.append(f"Input song not found: {input_path}")

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Song Cover Pipeline — 歌曲翻唱管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --config configs/ria_cover.yaml
  python run_pipeline.py --config configs/ria_cover.yaml --force
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/ria_cover.yaml",
        help="Path to YAML configuration file (default: configs/ria_cover.yaml)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-run all stages, ignoring checkpoint/resume state",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration, do not run the pipeline",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print("=" * 60)
    print("  Song Cover Pipeline — 歌曲翻唱管线")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Config:       {config_path}")
    print(f"  Force:        {args.force}")
    print("=" * 60)
    print()

    # Load and validate config
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        sys.exit(1)

    # Validate and report warnings
    warnings = validate_config(config)
    if warnings:
        logger.warning("Configuration issues detected:")
        for i, w in enumerate(warnings, 1):
            logger.warning("  %d. %s", i, w)
        print()

    if args.validate_only:
        if warnings:
            logger.info("Validation complete with %d warning(s).", len(warnings))
        else:
            logger.info("Configuration looks good!")
        return

    # Run pipeline
    from src.pipeline import SongCoverPipeline

    pipeline = SongCoverPipeline(config)

    # Handle --force by clearing checkpoints
    if args.force and pipeline._checkpoint_file.is_file():
        logger.info("--force: clearing checkpoint file")
        pipeline._checkpoint_file.unlink()
        pipeline._completed_stages = set()

    try:
        results = pipeline.run()

        print()
        print("=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)
        for stage_name, stage_result in results.get("stages", {}).items():
            status = stage_result.get("status", "unknown")
            files = stage_result.get("files", {})
            print(f"  [{status}] {stage_name}")
            for fname, fpath in files.items():
                print(f"         -> {fname}: {fpath}")
        print(f"  Total time: {results.get('elapsed', 0):.1f}s")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user. Checkpoint saved for resume.")
        sys.exit(130)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
