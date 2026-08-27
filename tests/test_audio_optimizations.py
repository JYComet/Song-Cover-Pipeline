import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_utils import compute_residuals
from src.audio_mixer import AudioMixer
from src.ddsp_converter import _SegmentStitcher, _cross_fade, _md5_file


class ResidualTests(unittest.TestCase):
    def test_direct_residuals_are_exact(self):
        rng = np.random.default_rng(7)
        original = rng.standard_normal((2, 4096), dtype=np.float32)
        vocals = rng.standard_normal((2, 4096), dtype=np.float32)
        dry = rng.standard_normal((2, 4096), dtype=np.float32)

        instrumental, reverb = compute_residuals(original, vocals, dry)

        np.testing.assert_array_equal(instrumental, original - vocals)
        np.testing.assert_array_equal(reverb, vocals - dry)

    def test_direct_residuals_expand_mono_without_changing_samples(self):
        stereo = np.arange(16, dtype=np.float32).reshape(2, 8)
        mono = np.arange(8, dtype=np.float32).reshape(1, 8)

        instrumental, reverb = compute_residuals(stereo, mono, mono * 0.5)

        expected_vocals = np.repeat(mono, 2, axis=0)
        np.testing.assert_array_equal(instrumental, stereo - expected_vocals)
        np.testing.assert_array_equal(reverb, mono - mono * 0.5)


class SegmentStitcherTests(unittest.TestCase):
    @staticmethod
    def _legacy_stitch(segments):
        result = np.zeros(0)
        current_length = 0
        for start, segment in segments:
            silent_length = start - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, segment)
            else:
                result = _cross_fade(result, segment, current_length + silent_length)
            current_length = current_length + silent_length + len(segment)
        return result

    def test_matches_legacy_for_gaps_and_cross_fades(self):
        rng = np.random.default_rng(11)
        segments = [
            (0, rng.standard_normal(10, dtype=np.float32)),
            (15, rng.standard_normal(8, dtype=np.float32)),
            (20, rng.standard_normal(9, dtype=np.float32)),
            (27, rng.standard_normal(12, dtype=np.float32)),
        ]
        expected = self._legacy_stitch(segments)

        stitcher = _SegmentStitcher(initial_capacity=4)
        for start, segment in segments:
            stitcher.add(start, segment)

        np.testing.assert_array_equal(stitcher.finish(), expected)

    def test_flattens_singleton_batch_dimension(self):
        stitcher = _SegmentStitcher()
        stitcher.add(0, np.arange(5, dtype=np.float32)[None, :])
        np.testing.assert_array_equal(stitcher.finish(), np.arange(5))


class HashingTests(unittest.TestCase):
    def test_streamed_md5_matches_existing_cache_key(self):
        payload = bytes(range(256)) * 100
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "audio.bin"
            path.write_bytes(payload)
            self.assertEqual(_md5_file(path, chunk_size=113), hashlib.md5(payload).hexdigest())


class MixerTests(unittest.TestCase):
    def test_direct_accumulation_matches_zero_padded_sum(self):
        vocal = np.arange(5, dtype=np.float32)[:, None] / 10
        instrumental = np.arange(14, dtype=np.float32).reshape(7, 2) / 20
        reverb = np.arange(3, dtype=np.float32)[:, None] / 30

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = [root / "vocal.wav", root / "instrumental.wav", root / "reverb.wav"]
            for path, samples in zip(paths, (vocal, instrumental, reverb)):
                sf.write(path, samples, 44100, format="WAV", subtype="FLOAT")

            output = root / "mix.wav"
            AudioMixer(reverb_gain_db=0.0, normalize_output=False).mix(*paths, output)
            actual, _ = sf.read(output, dtype="float32", always_2d=True)

        expected = np.zeros((7, 2), dtype=np.float32)
        expected[:5] += np.repeat(vocal, 2, axis=1)
        expected += instrumental
        expected[:3] += np.repeat(reverb, 2, axis=1)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
