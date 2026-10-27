"""Synthesised sound design for the videos (no samples needed).

    python3 sfx.py <out_dir> <bed_seconds>

Writes whoosh.wav, impact.wav, riser.wav and bed.wav (a dark 140 BPM pulse).
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
rng = np.random.default_rng(7)


def env(n, attack, release):
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a) ** 2
    e[n - r:] = np.linspace(1, 0, r) ** 2
    return e


def sweep_filter(x, f0, f1, kind='bandpass', blocks=64):
    # time-varying filter, approximated block by block
    out = np.zeros_like(x)
    edges = np.linspace(0, len(x), blocks + 1).astype(int)
    for i in range(blocks):
        f = f0 * (f1 / f0) ** (i / (blocks - 1))
        if kind == 'bandpass':
            sos = butter(2, [f * 0.6, min(f * 1.6, SR / 2 - 100)], 'bandpass', fs=SR, output='sos')
        else:
            sos = butter(2, f, 'lowpass', fs=SR, output='sos')
        seg = x[max(0, edges[i] - 2000):edges[i + 1]]
        out[edges[i]:edges[i + 1]] = sosfilt(sos, seg)[-(edges[i + 1] - edges[i]):]
    return out


def whoosh(d=0.55):
    n = int(d * SR)
    x = sweep_filter(rng.standard_normal(n), 300, 4500)
    t = np.linspace(0, 1, n)
    x *= np.exp(-((t - 0.62) / 0.22) ** 2)  # swell peaking just before the cut
    return x / np.abs(x).max() * 0.8


def impact(d=1.1):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 42 + 110 * np.exp(-t * 22)  # pitch-dropping sub thump
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    click = sosfilt(butter(2, 3500, 'lowpass', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-t * 60)
    tail = sweep_filter(rng.standard_normal(n), 2500, 300, 'lowpass') * np.exp(-t * 6) * 0.25
    x = np.tanh(1.8 * (body + 0.5 * click + tail))
    return x / np.abs(x).max() * 0.9


def riser(d=1.6):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    noise = sweep_filter(rng.standard_normal(n), 400, 8000)
    tone = np.sin(2 * np.pi * np.cumsum(180 + 900 * t ** 2) / SR) * 0.3
    x = (noise / np.abs(noise).max() + tone) * t ** 2.2
    x[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return x / np.abs(x).max() * 0.7


def bed(d, bpm=140):
    n = int(d * SR)
    t = np.arange(n) / SR
    beat = 60 / bpm
    x = np.zeros(n)
    # pad: A minor, detuned saws through a slowly breathing low-pass
    pad = np.zeros(n)
    for f in (55, 110, 130.81, 164.81, 220):
        for det in (-0.12, 0.12):
            ph = (t * f * (1 + det / 100)) % 1
            pad += (2 * ph - 1) / (1 + f / 110)
    pad = sweep_filter(pad, 500, 900, 'lowpass', blocks=16) * (0.8 + 0.2 * np.sin(2 * np.pi * t / 8))
    x += pad / np.abs(pad).max() * 0.35
    # kick on every beat, hat on the off-beats
    kick_n = int(0.35 * SR)
    kt = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-kt * 30)) / SR) * np.exp(-kt * 9)
    hat_n = int(0.05 * SR)
    hat = sosfilt(butter(2, 7000, 'highpass', fs=SR, output='sos'), rng.standard_normal(hat_n)) * np.exp(-np.arange(hat_n) / SR * 80)
    k = 0
    while k * beat < d:
        i = int(k * beat * SR)
        m = min(kick_n, n - i)
        x[i:i + m] += kick[:m] * 0.9
        j = int((k + 0.5) * beat * SR)
        if j < n:
            m = min(hat_n, n - j)
            x[j:j + m] += hat[:m] * 0.25
        k += 1
    x *= env(n, 0.8, 1.5)
    return np.tanh(x * 1.2) * 0.6


if __name__ == '__main__':
    out, seconds = sys.argv[1], float(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    for name, x in (('whoosh', whoosh()), ('impact', impact()), ('riser', riser()), ('bed', bed(seconds))):
        sf.write(os.path.join(out, name + '.wav'), x.astype(np.float32), SR)
