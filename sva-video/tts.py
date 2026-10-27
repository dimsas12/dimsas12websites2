"""Offline English voiceover (Kokoro v1.0 via sherpa-onnx).

    python3 tts.py <out.wav> "text to speak" [speed]

Prints JSON: duration, sentence start times and estimated per-word timings.

Text is spoken sentence by sentence with natural pauses in between;
"..." adds a dramatic beat. The voice is a blend of Kokoro speakers, set with
SVA_VOICE (default "am_puck:0.6,am_fenrir:0.4").

Model: kokoro-multi-lang-v1_0 from github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models,
expected in $SVA_TTS_DIR (default: ./tts-models).
"""
import json
import os
import re
import sys

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = os.environ.get('SVA_TTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts-models'))
MODEL = os.path.join(ROOT, 'kokoro-multi-lang-v1_0')
VOICE = os.environ.get('SVA_VOICE', 'am_puck:0.6,am_fenrir:0.4')
SPEED = float(os.environ.get('SVA_SPEED', '1.1'))
PAUSE = {'.': 0.16, '!': 0.16, '?': 0.22, '…': 0.38, ';': 0.12}
N_SPEAKERS, FRAMES, DIM = 54, 510, 256


def speaker_names():
    with open(f'{MODEL}/model.onnx', 'rb') as f:
        blob = f.read()
    return re.search(rb'af_alloy(?:,[a-z]+_[a-z]+)+', blob).group().decode().split(',')[:N_SPEAKERS]


def blended_model_dir():
    """Kokoro voices are style vectors, so a weighted mix makes a new voice.
    The mix is written into the last slot of a private copy of voices.bin."""
    out = os.path.join(ROOT, 'kokoro-blend-' + re.sub(r'[^a-z0-9]+', '_', VOICE))
    if not os.path.exists(f'{out}/voices.bin'):
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(MODEL):
            if name != 'voices.bin' and not os.path.exists(f'{out}/{name}'):
                os.symlink(f'{MODEL}/{name}', f'{out}/{name}')
        names = speaker_names()
        voices = np.fromfile(f'{MODEL}/voices.bin', dtype=np.float32).reshape(N_SPEAKERS, FRAMES, DIM)
        mix = np.zeros((FRAMES, DIM), dtype=np.float32)
        for part in VOICE.split(','):
            name, w = part.split(':')
            mix += float(w) * voices[names.index(name)]
        voices[-1] = mix
        voices.tofile(f'{out}/voices.bin')
    return out


def make():
    d = blended_model_dir()
    model = sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=f'{d}/model.onnx', voices=f'{d}/voices.bin', tokens=f'{d}/tokens.txt',
        data_dir=f'{d}/espeak-ng-data', dict_dir=f'{d}/dict',
        lexicon=f'{d}/lexicon-us-en.txt,{d}/lexicon-zh.txt', lang='en-us'),
        num_threads=4)
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model))


def trim(x, sr):
    loud = np.flatnonzero(np.abs(x) > 0.015)
    if not len(loud):
        return x
    pad = int(0.02 * sr)
    return x[max(0, loud[0] - pad):loud[-1] + pad]


def main():
    out, text = sys.argv[1:3]
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else SPEED
    tts = make()
    text = text.replace('...', '…')
    # split after sentence punctuation, keeping it for intonation
    parts = [p.strip() for p in re.split(r'(?<=[.!?…;])\s+', text) if p.strip()]
    chunks, sr, pos, cues, words = [], 24000, 0, [], []
    for p in parts:
        spoken = p.rstrip('…').rstrip() + ('.' if p.endswith('…') else '')
        audio = tts.generate(spoken, sid=N_SPEAKERS - 1, speed=speed)
        sr = audio.sample_rate
        x = trim(np.array(audio.samples, dtype=np.float32), sr)
        start, d = pos / sr, len(x) / sr
        cues.append(round(start, 3))
        # word timings are estimated by spreading the sentence over its letters
        ws = p.split()
        weights = [len(w) + 2 for w in ws]
        acc = 0
        for w, k in zip(ws, weights):
            words.append({'w': w, 't': round(start + d * acc / sum(weights), 3)})
            acc += k
        gap = np.zeros(int(PAUSE.get(p[-1], 0.1) * sr), dtype=np.float32)
        chunks += [x, gap]
        pos += len(x) + len(gap)
    samples = np.concatenate(chunks[:-1]) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(out, samples, sr)
    print(json.dumps({'dur': round(len(samples) / sr, 3), 'cues': cues, 'words': words}))


if __name__ == '__main__':
    main()
