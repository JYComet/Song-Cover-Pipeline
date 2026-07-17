"""
Song Cover Pipeline Runner

Orchestrates the full pipeline:
  1. Harmony separation  (MSST karaoke model)
  2. Reverb separation   (MSST dereverb model)
  3. Timbre conversion   (DDSP-SVC RectifiedFlow)
  4. Final mixing        (vocals + instrumental + reverb)

All parameters are driven by a YAML configuration file.
Each stage saves intermediate outputs, enabling resume from
any point if the pipeline is interrupted.
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


class SongCoverPipeline:
    """
    Complete song cover generation pipeline.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary (parsed from YAML).

    Usage
    -----
    >>> pipeline = SongCoverPipeline(config)
    >>> pipeline.run()
    """

    def __init__(self, config: dict):
        self.config = config

        # Resolve paths relative to the project root
        self.project_root = Path(config.get("_project_root", os.getcwd()))
        logger.info("Project root: %s", self.project_root)

        # Task info
        task = config.get("task", {})
        self.task_name = task.get("name", "unnamed")
        self.input_song = self._resolve(task.get("input_song", ""))
        self.output_dir = self._resolve(task.get("output_dir", "output/default"))

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage checkpoints file (for resume support)
        self._checkpoint_file = self.output_dir / ".pipeline_checkpoint.json"
        self._completed_stages = self._load_checkpoint()

        logger.info("=" * 60)
        logger.info("Pipeline: %s", self.task_name)
        logger.info("Input: %s", self.input_song)
        logger.info("Output dir: %s", self.output_dir)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Public: run the full pipeline
    # ------------------------------------------------------------------

    def run(self, progress_callback: Optional[Callable] = None) -> dict:
        """
        Execute all enabled pipeline stages in order.

        Parameters
        ----------
        progress_callback : callable or None
            Optional callback for progress reporting.
            Signature: callback(stage: str, status: str, percent: float, message: str)
            - stage: stage key (e.g. "harmony_separation", "timbre_conversion")
            - status: "started", "progress", "completed", "error"
            - percent: 0.0–100.0 overall progress
            - message: human-readable description

        Returns
        -------
        dict
            Summary of results including output file paths and timings.
        """
        def _progress(stage, status, percent, message):
            if progress_callback:
                try:
                    progress_callback(stage, status, percent, message)
                except Exception:
                    pass  # never let callback failures break the pipeline

        results = {
            "task_name": self.task_name,
            "stages": {},
            "start_time": time.time(),
        }

        try:
            # Stage 0: Extract audio from video (if needed)
            if self._stage_enabled("extract_audio"):
                _progress("extract_audio", "started", 0.0, "Extracting audio from input...")
                results["stages"]["extract_audio"] = self._run_extract_audio()
                _progress("extract_audio", "completed", 5.0, "Audio extraction complete")

            # Check if linked separation is enabled (runs karaoke+dereverb in one shot)
            linked_cfg = self.config.get("linked_separation", {})
            if linked_cfg.get("enabled", False):
                # Stages 1+2: Linked karaoke → dereverb (GPU memory pass-through)
                _progress("linked_separation", "started", 5.0,
                          "加载和声分离模型 (~200MB), 首次运行需稍等...")
                linked_result = self._run_linked_separation(progress_callback=progress_callback)
                _progress("linked_separation", "completed", 45.0, "和声+混响分离完成")
                results["stages"]["linked_separation"] = linked_result
                results["stages"]["harmony_separation"] = {
                    "status": "completed (via linked)",
                    "files": {k: str(v) for k, v in linked_result.get("files", {}).items()
                              if k in ("Vocals", "Instrumental")},
                }
                results["stages"]["reverb_separation"] = {
                    "status": "completed (via linked)",
                    "files": {k: str(v) for k, v in linked_result.get("files", {}).items()
                              if k in ("noreverb", "reverb")},
                }
            else:
                # Stage 1: Harmony separation (standalone)
                if self._stage_enabled("harmony_separation"):
                    _progress("harmony_separation", "started", 5.0,
                              "加载和声分离模型 (~200MB), 首次运行需稍等...")
                    results["stages"]["harmony_separation"] = self._run_harmony_separation()
                    _progress("harmony_separation", "completed", 25.0, "和声分离完成, 人声+伴奏已分离")

                # Stage 2: Reverb separation (standalone)
                if self._stage_enabled("reverb_separation"):
                    _progress("reverb_separation", "started", 25.0,
                              "加载混响分离模型 (~200MB), 首次运行需稍等...")
                    results["stages"]["reverb_separation"] = self._run_reverb_separation()
                    _progress("reverb_separation", "completed", 45.0, "混响分离完成, 干声+混响尾已分离")

            # Stage 3: Timbre conversion
            if self._stage_enabled("timbre_conversion"):
                _progress("timbre_conversion", "started", 45.0,
                          "加载DDSP音色模型 (~210MB), 首次运行需稍等...")
                results["stages"]["timbre_conversion"] = self._run_timbre_conversion(
                    progress_callback=progress_callback
                )
                _progress("timbre_conversion", "completed", 90.0, "音色替换完成!")

            # Stage 4: Final mix
            if self._stage_enabled("mixing"):
                _progress("mixing", "started", 90.0, "Final mixing...")
                results["stages"]["mixing"] = self._run_mixing()
                _progress("mixing", "completed", 100.0, "Pipeline complete!")

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            results["error"] = str(exc)
            _progress("error", "error", -1, str(exc))
            raise

        finally:
            results["end_time"] = time.time()
            results["elapsed"] = results["end_time"] - results["start_time"]
            logger.info(
                "Pipeline completed in %.1f seconds (%.1f minutes)",
                results["elapsed"],
                results["elapsed"] / 60,
            )
            self._save_checkpoint()

        return results

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    # Supported video extensions for audio extraction
    _VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v", ".m4a"}

    def _run_extract_audio(self) -> dict:
        """Stage 0: Extract audio from video if the input is a video file.

        If the input is already an audio file, this stage is a no-op.
        If it's a video, use ffmpeg to extract the audio track as WAV.
        """
        stage_cfg = self.config.get("extract_audio", {})
        stage_name = "00_extract_audio"

        input_path = self.input_song

        # Check if input is a video file
        suffix = input_path.suffix.lower()
        if suffix not in self._VIDEO_EXTS:
            logger.info(
                "Stage 0 (extract audio): input is already audio (%s), skipping.",
                suffix,
            )
            return {"status": "skipped", "reason": f"input is audio ({suffix})"}

        logger.info("=" * 50)
        logger.info("STAGE 0: Extract Audio from Video")
        logger.info("=" * 50)
        logger.info("Input video: %s", input_path.name)

        stage_dir = self.output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        output_wav = stage_dir / f"{input_path.stem}.wav"
        sample_rate = stage_cfg.get("sample_rate", 44100)

        if output_wav.is_file():
            logger.info("Audio already extracted: %s", output_wav.name)
        else:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-acodec", "pcm_f32le",
                "-ar", str(sample_rate),
                "-ac", "2",
                str(output_wav),
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg audio extraction failed:\n{result.stderr}"
                )
            logger.info(
                "Audio extracted: %s (%.1fs, %d Hz)",
                output_wav.name,
                float(subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(output_wav)]
                ).decode().strip()),
                sample_rate,
            )

        # Update the input song path for downstream stages
        self.input_song = output_wav

        result = {
            "status": "completed",
            "output_dir": str(stage_dir),
            "files": {"extracted_audio": str(output_wav)},
        }
        return result

    def _run_harmony_separation(self) -> dict:
        """Stage 1: Separate lead vocals from instrumental/harmony."""
        stage_cfg = self.config["harmony_separation"]
        stage_name = "01_harmony_separation"

        if self._is_stage_done(stage_name):
            logger.info("Stage 1 (harmony separation): already completed, skipping.")
            return {"status": "skipped", "output_dir": str(self.output_dir / stage_name)}

        logger.info("=" * 50)
        logger.info("STAGE 1: Harmony Separation (和声分离)")
        logger.info("=" * 50)

        from src.msst_separator import MSSTSeparator

        separator = MSSTSeparator(
            msst_code_dir=stage_cfg["msst_code_dir"],
            model_type=stage_cfg.get("model_type", "bs_roformer"),
            config_path=stage_cfg["config_path"],
            checkpoint_path=stage_cfg["checkpoint_path"],
            device=stage_cfg.get("device", "cuda:0"),
            chunk_batch=stage_cfg.get("chunk_batch", 16),
            use_compile=stage_cfg.get("use_compile", True),
        )

        try:
            separator.load_model()

            target_stem = stage_cfg.get("target_stem", "Vocals")
            other_stems = stage_cfg.get("other_stems", ["Instrumental"])

            stage_dir = self.output_dir / stage_name
            saved = separator.separate_to_file(
                input_path=self.input_song,
                output_dir=stage_dir,
                target_stem=target_stem,
                other_stems=other_stems,
                output_sample_rate=stage_cfg.get("output_sample_rate", 44100),
            )

            # If the model only outputs the target stem (e.g. Vocals) but we
            # also want the complementary stem (e.g. Instrumental), compute it
            # as the residual: complementary = original_input - target
            for requested in other_stems:
                if requested not in saved:
                    _compute_residual_stem(
                        input_file=self.input_song,
                        target_file=saved[target_stem],
                        residual_name=requested,
                        output_dir=stage_dir,
                    )
                    residual_path = _find_residual_file(stage_dir, requested)
                    if residual_path:
                        saved[requested] = residual_path
                        logger.info(
                            "Derived residual stem: %s = input - %s",
                            requested, target_stem,
                        )

            result = {
                "status": "completed",
                "output_dir": str(stage_dir),
                "files": {k: str(v) for k, v in saved.items()},
            }
            self._mark_stage_done(stage_name)
            return result

        finally:
            separator.unload_model()

    def _run_reverb_separation(self) -> dict:
        """Stage 2: Separate dry vocals from reverb tail."""
        stage_cfg = self.config["reverb_separation"]
        stage_name = "02_reverb_separation"

        if self._is_stage_done(stage_name):
            logger.info("Stage 2 (reverb separation): already completed, skipping.")
            return {"status": "skipped", "output_dir": str(self.output_dir / stage_name)}

        logger.info("=" * 50)
        logger.info("STAGE 2: Reverb Separation (混响分离)")
        logger.info("=" * 50)

        # Find the vocals file from Stage 1
        vocals_file = self._find_stage_output(
            "01_harmony_separation", "Vocals", "vocals"
        )
        if vocals_file is None:
            raise FileNotFoundError(
                "Cannot find vocals output from Stage 1. "
                "Harmony separation must be run first."
            )
        logger.info("Input vocals: %s", vocals_file.name)

        from src.msst_separator import MSSTSeparator

        separator = MSSTSeparator(
            msst_code_dir=stage_cfg["msst_code_dir"],
            model_type=stage_cfg.get("model_type", "bs_roformer"),
            config_path=stage_cfg["config_path"],
            checkpoint_path=stage_cfg["checkpoint_path"],
            device=stage_cfg.get("device", "cuda:0"),
            chunk_batch=stage_cfg.get("chunk_batch", 8),
            use_compile=stage_cfg.get("use_compile", True),
        )

        try:
            separator.load_model()

            target_stem = stage_cfg.get("target_stem", "noreverb")
            other_stems = stage_cfg.get("other_stems", ["reverb"])

            stage_dir = self.output_dir / stage_name
            saved = separator.separate_to_file(
                input_path=vocals_file,
                output_dir=stage_dir,
                target_stem=target_stem,
                other_stems=other_stems,
                output_sample_rate=stage_cfg.get("output_sample_rate", 44100),
            )

            # If the model only outputs the target stem (e.g. noreverb) but
            # we also want the complementary stem (e.g. reverb), compute it
            # as the residual: complementary = original_input - target
            for requested in other_stems:
                if requested not in saved:
                    _compute_residual_stem(
                        input_file=vocals_file,
                        target_file=saved[target_stem],
                        residual_name=requested,
                        output_dir=stage_dir,
                    )
                    residual_path = _find_residual_file(stage_dir, requested)
                    if residual_path:
                        saved[requested] = residual_path
                        logger.info(
                            "Derived residual stem: %s = input - %s",
                            requested, target_stem,
                        )

            result = {
                "status": "completed",
                "output_dir": str(stage_dir),
                "files": {k: str(v) for k, v in saved.items()},
            }
            self._mark_stage_done(stage_name)
            return result

        finally:
            separator.unload_model()

    def _run_linked_separation(self, progress_callback: Optional[Callable] = None) -> dict:
        """
        Stages 1+2 combined: Karaoke → Dereverb with in-memory data passing.

        Instead of loading both models simultaneously (which causes GPU memory
        bandwidth contention), we use a STAGED approach:
          1. Load karaoke → run → keep vocals in RAM → UNLOAD karaoke
          2. Load dereverb → run on in-memory vocals → UNLOAD dereverb
          3. Compute residuals in STFT domain (fast + precise)

        This avoids:
          - Disk I/O for intermediate vocals.wav
          - GPU memory bandwidth contention (only 1 model on GPU at a time)
          - librosa-based residual computation (replaced by STFT-domain)
        """
        def _p(stage, status, pct, msg):
            if progress_callback:
                try:
                    progress_callback(stage, status, pct, msg)
                except Exception:
                    pass
        karaoke_cfg = self.config["harmony_separation"]
        dereverb_cfg = self.config["reverb_separation"]

        if self._is_stage_done("01_harmony_separation") and self._is_stage_done("02_reverb_separation"):
            logger.info("Linked separation: both stages already completed, skipping.")
            return {"status": "skipped"}

        logger.info("=" * 50)
        logger.info("STAGES 1+2: Linked Separation (和声分离 + 混响分离)")
        logger.info("=" * 50)

        from src.msst_separator import MSSTSeparator

        # Load input audio
        import soundfile as sf
        import numpy as np
        audio, sr = sf.read(str(self.input_song))
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        else:
            audio = audio.T  # (samples, ch) → (ch, samples)
        audio = audio.astype(np.float32)
        sample_rate_out = karaoke_cfg.get("output_sample_rate", 44100)

        # Resample original audio to target rate if needed (for STFT residual alignment)
        if sr != sample_rate_out:
            logger.info("Resampling original from %d Hz to %d Hz", sr, sample_rate_out)
            from src.msst_separator import _resample_fast
            audio = _resample_fast(audio, sr, sample_rate_out)
            sr = sample_rate_out
        stage_dir_1 = self.output_dir / "01_harmony_separation"
        stage_dir_1.mkdir(parents=True, exist_ok=True)
        base_name = self.input_song.stem

        # ==================================================================
        # Phase 1: Karaoke separation → keep vocals in memory
        # ==================================================================
        logger.info("--- Phase 1: Karaoke separation ---")
        karaoke = MSSTSeparator(
            msst_code_dir=karaoke_cfg["msst_code_dir"],
            model_type=karaoke_cfg.get("model_type", "bs_roformer"),
            config_path=karaoke_cfg["config_path"],
            checkpoint_path=karaoke_cfg["checkpoint_path"],
            device=karaoke_cfg.get("device", "cuda:0"),
            chunk_batch=karaoke_cfg.get("chunk_batch", 16),
            use_compile=karaoke_cfg.get("use_compile", True),
        )
        try:
            karaoke.load_model()
            _p("linked_separation", "progress", 8.0, "和声分离中...")
            karaoke_waveforms = karaoke.separate(audio, sr, "Vocals")
            vocals = karaoke_waveforms["Vocals"]  # numpy (ch, samples) — kept in RAM
        finally:
            karaoke.unload_model()

        # Save vocals and compute instrumental from STFT
        vocals_path = stage_dir_1 / f"{base_name}_Vocals.wav"
        sf.write(str(vocals_path), vocals.T, sample_rate_out, format="WAV", subtype="FLOAT")
        logger.info("Saved: %s", vocals_path.name)

        # ==================================================================
        # Phase 2: Dereverb separation on in-memory vocals
        # ==================================================================
        _p("linked_separation", "progress", 15.0, "加载混响分离模型 (~200MB)...")
        logger.info("--- Phase 2: Dereverb separation (in-memory) ---")
        dereverb = MSSTSeparator(
            msst_code_dir=dereverb_cfg["msst_code_dir"],
            model_type=dereverb_cfg.get("model_type", "bs_roformer"),
            config_path=dereverb_cfg["config_path"],
            checkpoint_path=dereverb_cfg["checkpoint_path"],
            device=dereverb_cfg.get("device", "cuda:0"),
            chunk_batch=dereverb_cfg.get("chunk_batch", 8),
            use_compile=dereverb_cfg.get("use_compile", True),
        )
        try:
            dereverb.load_model()
            _p("linked_separation", "progress", 25.0, "混响分离中...")
            dereverb_waveforms = dereverb.separate(vocals, sr, "noreverb")
            noreverb = dereverb_waveforms["noreverb"]  # numpy (ch, samples)
        finally:
            dereverb.unload_model()

        # Save noreverb
        reverb_dir = self.output_dir / "02_reverb_separation"
        reverb_dir.mkdir(parents=True, exist_ok=True)
        noreverb_path = reverb_dir / f"{base_name}_Vocals_noreverb.wav"
        sf.write(str(noreverb_path), noreverb.T, sample_rate_out, format="WAV", subtype="FLOAT")
        logger.info("Saved: %s", noreverb_path.name)

        # ==================================================================
        # Phase 3: Residual computation in STFT domain (fast + precise)
        # ==================================================================
        logger.info("--- Phase 3: STFT-domain residual computation ---")
        instrumental, reverb = _compute_residuals_stft(
            audio, vocals, noreverb, sr,
            n_fft=2048, hop_length=512, win_length=2048,
        )

        instrumental_path = stage_dir_1 / f"{base_name}_Instrumental.wav"
        sf.write(str(instrumental_path), instrumental.T, sample_rate_out, format="WAV", subtype="FLOAT")
        logger.info("Saved: %s", instrumental_path.name)

        reverb_path = reverb_dir / f"{base_name}_Vocals_reverb.wav"
        sf.write(str(reverb_path), reverb.T, sample_rate_out, format="WAV", subtype="FLOAT")
        logger.info("Saved: %s", reverb_path.name)

        result = {
            "status": "completed",
            "files": {
                "Vocals": str(vocals_path),
                "noreverb": str(noreverb_path),
                "Instrumental": str(instrumental_path),
                "reverb": str(reverb_path),
            },
        }
        self._mark_stage_done("01_harmony_separation")
        self._mark_stage_done("02_reverb_separation")
        return result

    def _run_timbre_conversion(self, progress_callback: Optional[Callable] = None) -> dict:
        """Stage 3: Convert vocal timbre using DDSP-SVC."""
        stage_cfg = self.config["timbre_conversion"]
        stage_name = "03_timbre_conversion"

        if self._is_stage_done(stage_name):
            logger.info("Stage 3 (timbre conversion): already completed, skipping.")
            return {"status": "skipped", "output_dir": str(self.output_dir / stage_name)}

        logger.info("=" * 50)
        logger.info("STAGE 3: Timbre Conversion (音色替换)")
        logger.info("=" * 50)

        # Find the dry vocals file from Stage 2
        dry_vocals_file = self._find_stage_output(
            "02_reverb_separation", "noreverb", "dry"
        )
        if dry_vocals_file is None:
            raise FileNotFoundError(
                "Cannot find dry vocals (noreverb) output from Stage 2. "
                "Reverb separation must be run first."
            )
        logger.info("Input dry vocals: %s", dry_vocals_file.name)

        from src.ddsp_converter import DDSPConverter

        converter = DDSPConverter(
            ddsp_project_dir=stage_cfg["ddsp_project_dir"],
            model_ckpt=stage_cfg["model_ckpt"],
            device=stage_cfg.get("device", "cuda:0"),
            infer_step=stage_cfg.get("infer_step", 100),
            t_start=stage_cfg.get("t_start", 0.4),
            method=stage_cfg.get("method", "rk4"),
            pitch_extractor=stage_cfg.get("pitch_extractor", "fcpe"),
            key=stage_cfg.get("key", 0),
            vocal_register_shift=stage_cfg.get("vocal_register_shift", 0),
            formant_shift=stage_cfg.get("formant_shift", 0),
            f0_min=stage_cfg.get("f0_min", 65),
            f0_max=stage_cfg.get("f0_max", 800),
            threshold=stage_cfg.get("threshold", -60),
            spk_id=stage_cfg.get("spk_id", 1),
            # Performance tuning
            use_compile=stage_cfg.get("use_compile", False),   # DDSP: crashes on this system
            use_amp=stage_cfg.get("use_amp", False),
            segment_batch_size=stage_cfg.get("segment_batch_size", 1),
        )

        try:
            converter.load_model()

            stage_dir = self.output_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            output_file = stage_dir / f"{dry_vocals_file.stem}_converted.wav"

            converter.convert(
                input_path=dry_vocals_file,
                output_path=output_file,
                progress_callback=progress_callback,
            )

            result = {
                "status": "completed",
                "output_dir": str(stage_dir),
                "files": {"converted_vocals": str(output_file)},
            }
            self._mark_stage_done(stage_name)
            return result

        finally:
            converter.unload_model()

    def _run_mixing(self) -> dict:
        """Stage 4: Mix all tracks into final cover song."""
        stage_cfg = self.config["mixing"]
        stage_name = "04_final_mix"

        # Mixing is fast, always run it
        logger.info("=" * 50)
        logger.info("STAGE 4: Final Mix (混音叠加)")
        logger.info("=" * 50)

        # Find all required tracks
        converted_vocals = self._find_stage_output(
            "03_timbre_conversion", "converted", "converted"
        )
        instrumental = self._find_stage_output(
            "01_harmony_separation", "Instrumental", "instrumental"
        )
        reverb = self._find_stage_output(
            "02_reverb_separation", "reverb", "reverb"
        )

        missing = []
        if converted_vocals is None:
            missing.append("converted vocals (Stage 3)")
        if instrumental is None:
            missing.append("instrumental (Stage 1)")
        if reverb is None:
            missing.append("reverb (Stage 2)")

        if missing:
            raise FileNotFoundError(
                "Missing tracks for mixing: " + ", ".join(missing)
            )

        logger.info("Converted vocals: %s", converted_vocals.name)
        logger.info("Instrumental:     %s", instrumental.name)
        logger.info("Reverb:           %s", reverb.name)

        from src.audio_mixer import AudioMixer

        mixer = AudioMixer(
            sample_rate=stage_cfg.get("output_sample_rate", 44100),
            vocal_gain_db=stage_cfg.get("vocal_gain", 0.0),
            instrumental_gain_db=stage_cfg.get("instrumental_gain", 0.0),
            reverb_gain_db=stage_cfg.get("reverb_gain", -3.0),
            normalize_output=stage_cfg.get("normalize_output", True),
            output_format=stage_cfg.get("output_format", "wav"),
        )

        stage_dir = self.output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename from input song name
        input_stem = Path(self.config["task"]["input_song"]).stem
        output_file = stage_dir / f"{input_stem}_cover.wav"

        mixer.mix(
            vocal_path=converted_vocals,
            instrumental_path=instrumental,
            reverb_path=reverb,
            output_path=output_file,
        )

        result = {
            "status": "completed",
            "output_dir": str(stage_dir),
            "files": {"final_mix": str(output_file)},
        }
        self._mark_stage_done(stage_name)
        return result

    # ------------------------------------------------------------------
    # Checkpoint / resume support
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> set:
        """Load set of completed stage names from checkpoint file."""
        if self._checkpoint_file.is_file():
            try:
                data = json.loads(self._checkpoint_file.read_text())
                completed = set(data.get("completed_stages", []))
                logger.info(
                    "Resume checkpoint found: %d stage(s) completed",
                    len(completed),
                )
                return completed
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt checkpoint file, starting fresh.")
        return set()

    def _save_checkpoint(self) -> None:
        """Save completed stages to checkpoint file."""
        data = {
            "task_name": self.task_name,
            "completed_stages": sorted(self._completed_stages),
        }
        self._checkpoint_file.write_text(json.dumps(data, indent=2))

    def _is_stage_done(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return stage_name in self._completed_stages

    def _mark_stage_done(self, stage_name: str) -> None:
        """Mark a stage as completed and persist."""
        self._completed_stages.add(stage_name)
        self._save_checkpoint()
        logger.info("Stage %s marked as completed.", stage_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage_enabled(self, stage_key: str) -> bool:
        """Check if a stage is enabled in config."""
        stage_cfg = self.config.get(stage_key, {})
        return stage_cfg.get("enabled", True)

    def _resolve(self, path: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_root / p

    def _find_stage_output(
        self, stage_dir_name: str, keyword: str, label: str
    ) -> Optional[Path]:
        """
        Find an output file from a previous stage by keyword matching.

        Searches the stage output directory for files containing `keyword`
        (case-insensitive) in their filename. Returns the first match.

        Parameters
        ----------
        stage_dir_name : str
            Subdirectory name under output_dir.
        keyword : str
            Case-insensitive substring to match in filenames.
        label : str
            Human-readable label for logging.

        Returns
        -------
        Path or None
        """
        stage_dir = self.output_dir / stage_dir_name
        if not stage_dir.is_dir():
            logger.warning("Stage directory not found: %s", stage_dir)
            return None

        wav_files = sorted(stage_dir.glob("*.wav"))
        if not wav_files:
            logger.warning("No WAV files in: %s", stage_dir)
            return None

        # Try to find a file matching the keyword.
        # Use suffix-based matching to avoid substring false positives
        # (e.g. "reverb" should NOT match "noreverb").
        keyword_lower = keyword.lower()
        exact_matches = [
            f for f in wav_files
            if f"_%s.wav" % keyword_lower in f.name.lower()
            or f.stem.lower().endswith("_" + keyword_lower)
        ]
        if exact_matches:
            logger.info("Found %s track: %s", label, exact_matches[0].name)
            return exact_matches[0]

        # Fallback: substring match (less precise)
        for f in wav_files:
            if keyword_lower in f.name.lower():
                logger.info("Found %s track (substring): %s", label, f.name)
                return f

        # Last resort: return the first WAV file
        logger.warning(
            "No file matching %r found in %s, using first: %s",
            keyword, stage_dir_name, wav_files[0].name,
        )
        return wav_files[0]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _compute_residual_stem(
    input_file: Path,
    target_file: Path,
    residual_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    Compute a residual stem: residual = input - target.

    Useful when a model outputs only one stem (e.g. noreverb) but we
    also want the complement (e.g. reverb = original - noreverb).

    Parameters
    ----------
    input_file : Path
        Original input to the separation model.
    target_file : Path
        Target stem produced by the model.
    residual_name : str
        Name for the residual stem (e.g. "reverb").
    output_dir : Path
        Directory to write the residual file.

    Returns
    -------
    Path or None
    """
    import numpy as np
    import librosa
    import soundfile as sf

    try:
        input_audio, sr = librosa.load(str(input_file), sr=None, mono=False)
        target_audio, _ = librosa.load(str(target_file), sr=None, mono=False)

        # Normalize shapes to (channels, samples)
        if input_audio.ndim == 1:
            input_audio = np.expand_dims(input_audio, axis=0)
        if target_audio.ndim == 1:
            target_audio = np.expand_dims(target_audio, axis=0)

        # Match channels
        if input_audio.shape[0] != target_audio.shape[0]:
            if input_audio.shape[0] == 1:
                input_audio = np.repeat(input_audio, target_audio.shape[0], axis=0)
            elif target_audio.shape[0] == 1:
                target_audio = np.repeat(target_audio, input_audio.shape[0], axis=0)

        # Align lengths
        min_len = min(input_audio.shape[-1], target_audio.shape[-1])
        input_audio = input_audio[:, :min_len]
        target_audio = target_audio[:, :min_len]

        # Compute residual
        residual = input_audio - target_audio

        # Write output
        stem = input_file.stem
        out_path = output_dir / f"{stem}_{residual_name}.wav"
        sf.write(str(out_path), residual.T, sr, format="WAV", subtype="FLOAT")
        return out_path

    except Exception as exc:
        logger.warning("Failed to compute residual stem %r: %s", residual_name, exc)
        return None


def _find_residual_file(output_dir: Path, residual_name: str) -> Optional[Path]:
    """Find a residual stem file by name suffix."""
    candidates = sorted(output_dir.glob(f"*_{residual_name}.wav"))
    return candidates[0] if candidates else None


def _compute_residuals_stft(
    original: "np.ndarray",
    vocals: "np.ndarray",
    noreverb: "np.ndarray",
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
) -> tuple:
    """
    Compute instrumental and reverb residuals in the STFT domain.

    instrumental = original - vocals   (complex STFT domain)
    reverb       = vocals - noreverb   (complex STFT domain)

    This is faster than time-domain subtraction because it avoids:
      - librosa.load overhead for large WAV files
      - Channel/length alignment logic
      - Precision loss from resampling

    Parameters
    ----------
    original : np.ndarray
        Original audio (channels, samples).
    vocals : np.ndarray
        Vocals stem (channels, samples).
    noreverb : np.ndarray
        Dry vocals stem (channels, samples).
    sample_rate : int
        Sample rate (unused, accepted for interface compatibility).
    n_fft, hop_length, win_length : int
        STFT parameters.

    Returns
    -------
    tuple of (instrumental, reverb) as numpy arrays.
    """
    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = torch.hann_window(win_length, device=device)

    def _stft(audio_np):
        """Compute STFT for all channels → list of (freq, time) complex."""
        audio_t = torch.from_numpy(audio_np.astype(np.float32)).to(device)
        stfts = []
        for ch in range(audio_t.shape[0]):
            s = torch.stft(
                audio_t[ch],
                n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )
            stfts.append(s)
        return stfts

    def _istft(stft_list, length):
        """iSTFT for all channels → (channels, samples) tensor."""
        audios = []
        for s in stft_list:
            a = torch.istft(
                s, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, length=length,
            )
            audios.append(a)
        return torch.stack(audios, dim=0)

    stft_orig = _stft(original)
    stft_voc = _stft(vocals)
    stft_dry = _stft(noreverb)

    # Align to minimum time frames
    min_frames = min(s.shape[-1] for s in stft_orig + stft_voc + stft_dry)
    stft_orig = [s[..., :min_frames] for s in stft_orig]
    stft_voc = [s[..., :min_frames] for s in stft_voc]
    stft_dry = [s[..., :min_frames] for s in stft_dry]

    # Residuals in STFT domain (complex subtraction)
    stft_inst = [o - v for o, v in zip(stft_orig, stft_voc)]
    stft_rev = [v - d for v, d in zip(stft_voc, stft_dry)]

    instrumental = _istft(stft_inst, original.shape[-1]).cpu().numpy()
    reverb = _istft(stft_rev, vocals.shape[-1]).cpu().numpy()
    return instrumental, reverb
