# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import MediaIO

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    # lazy import scipy.signal, otherwise it will crash doc build.
    import scipy.signal

    if orig_sr > target_sr:
        return scipy.signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy.signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


def _sinc_interp_hann_resample(
    samples: npt.NDArray[np.floating],
    orig_sr: float,
    target_sr: float,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> npt.NDArray[np.floating]:
    """Port of torchaudio.functional.resample (default sinc_interp_hann)."""
    import torch
    import torch.nn.functional as F

    gcd = math.gcd(int(orig_sr), int(target_sr))
    orig_freq = int(orig_sr) // gcd
    new_freq = int(target_sr) // gcd

    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)

    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None]
    idx = idx / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=torch.float64)[:, None, None] / new_freq
    t = (t + idx) * base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2)**2
    t = t * math.pi
    kernels = torch.where(t == 0, torch.tensor(1.0, dtype=t.dtype), t.sin() / t)
    kernels = kernels * window * (base_freq / orig_freq)
    kernels = kernels.to(torch.float32).view(new_freq, 1, -1)

    wave_t = torch.from_numpy(np.asarray(samples, dtype=np.float32)).view(1, -1)
    length = wave_t.shape[-1]
    padded = F.pad(wave_t, (width, width + orig_freq))
    out = F.conv1d(padded[:, None], kernels, stride=orig_freq)
    out = out.transpose(1, 2).reshape(1, -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    return out[0, :target_length].numpy()


def resample_audio_torchaudio(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    """Match LFM2.5 prepare_audio_encoder_input.py / ChatState resampling."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)

    # Board torchaudio wheels may be linked to libcudart and fail to load
    # even when torch itself is fine. Fall back to the same sinc kernel.
    try:
        import torch
        import torchaudio

        wave_t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        wave_t = torchaudio.functional.resample(wave_t, int(orig_sr), int(target_sr))
        return wave_t.squeeze(0).numpy()
    except (ImportError, OSError):
        return _sinc_interp_hann_resample(audio, orig_sr, target_sr)


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: Optional[float] = None,
        method: Literal["librosa", "scipy", "torchaudio"] = "librosa",
    ):
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError("Audio resampling is not supported when "
                               "`target_sr` is not provided")
        if self.method == "librosa":
            return resample_audio_librosa(audio,
                                          orig_sr=orig_sr,
                                          target_sr=self.target_sr)
        elif self.method == "scipy":
            return resample_audio_scipy(audio,
                                        orig_sr=orig_sr,
                                        target_sr=self.target_sr)
        elif self.method == "torchaudio":
            return resample_audio_torchaudio(audio,
                                             orig_sr=orig_sr,
                                             target_sr=self.target_sr)
        else:
            raise ValueError(
                f"Invalid resampling method: {self.method}. "
                "Supported methods are 'librosa', 'scipy', and 'torchaudio'.")


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):

    @staticmethod
    def _load_pcm16_wav(data: bytes) -> Optional[tuple[npt.NDArray, float]]:
        """Match LFM prepare_audio_encoder_input.py: stdlib wave + int16/32768.

        Returns None when the payload is not a 16-bit PCM WAV so callers can
        fall back to librosa for other formats.
        """
        import wave

        try:
            with wave.open(BytesIO(data), "rb") as handle:
                if handle.getsampwidth() != 2:
                    return None
                orig_sr = handle.getframerate()
                nch = handle.getnchannels()
                raw = handle.readframes(handle.getnframes())
        except wave.Error:
            return None

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if nch > 1:
            samples = samples.reshape(-1, nch).mean(axis=1)
        return samples, float(orig_sr)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        pcm = self._load_pcm16_wav(data)
        if pcm is not None:
            return pcm
        return librosa.load(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return self.load_bytes(Path(filepath).read_bytes())

    def encode_base64(self, media: tuple[npt.NDArray, float]) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')
