"""Offline English voiceover (Kokoro v1.0).

    python3 tts.py <out.wav> "text to speak" [speed]

Prints JSON: duration, sentence start times and estimated per-word timings.

Text is spoken sentence by sentence with natural pauses in between;
"..." adds a dramatic beat. The voice is a blend of Kokoro speakers, set with
SVA_VOICE (default "am_puck:0.6,am_fenrir:0.4").

Names are pronounced from pronounce.json (word -> IPA): espeak phonemizes the
rest of the sentence, our IPA is spliced in for each name, and the Kokoro model
is run directly on those phonemes, so names come out right with no hiccups.

Model: kokoro-multi-lang-v1_0 from github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models,
expected in $SVA_TTS_DIR (default: ./tts-models).
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('SVA_TTS_DIR', os.path.join(HERE, 'tts-models'))
MODEL = os.path.join(ROOT, 'kokoro-multi-lang-v1_0')
VOICE = os.environ.get('SVA_VOICE', 'am_puck:0.6,am_fenrir:0.4')
SPEED = float(os.environ.get('SVA_SPEED', '1.1'))
PAUSE = {'.': 0.16, '!': 0.16, '?': 0.22, '…': 0.38, ';': 0.12}
N_SPEAKERS, FRAMES, DIM = 54, 510, 256
SR = 24000
LEAD_IN = os.environ.get('SVA_LEAD_IN', '… ')

# Runs sherpa-onnx's espeak frontend (debug mode logs the token ids) to turn
# plain text into Kokoro phoneme tokens.
PHONEMIZE = r'''
import sys, sherpa_onnx
d = sys.argv[1]
m = sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
    model=f'{d}/model.onnx', voices=f'{d}/voices.bin', tokens=f'{d}/tokens.txt', data_dir=f'{d}/espeak-ng-data',
    dict_dir=f'{d}/dict', lexicon=f'{d}/lexicon-us-en.txt,{d}/lexicon-zh.txt', lang='en-us'), num_threads=2, debug=True)
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=m))
for line in sys.argv[2:]:
    tts.generate(line, sid=0, speed=1.0)
'''


def load_tokens():
    sym2id = {}
    for line in open(f'{MODEL}/tokens.txt', encoding='utf-8'):
        line = line.rstrip('\n')
        if line:
            sym, i = line[:line.rindex(' ')], int(line[line.rindex(' ') + 1:])
            sym2id[sym or ' '] = i
    return sym2id


def phonemize(lines):
    r = subprocess.run([sys.executable, '-c', PHONEMIZE, MODEL] + lines, capture_output=True, text=True)
    ids = re.findall(r'ConvertTextToTokenIds:\d+ \n([0-9 ]+)\n', r.stderr)
    if len(ids) != len(lines):
        raise RuntimeError('phonemizer failed:\n' + r.stderr[-2000:])
    return [[int(x) for x in s.split()][1:-1] for s in ids]   # drop the boundary 0s


def speaker_names():
    with open(f'{MODEL}/model.onnx', 'rb') as f:
        blob = f.read()
    return re.search(rb'af_alloy(?:,[a-z]+_[a-z]+)+', blob).group().decode().split(',')[:N_SPEAKERS]


def blended_voice():
    """Kokoro voices are style vectors, so a weighted mix makes a new voice."""
    names = speaker_names()
    voices = np.fromfile(f'{MODEL}/voices.bin', dtype=np.float32).reshape(N_SPEAKERS, FRAMES, DIM)
    mix = np.zeros((FRAMES, DIM), dtype=np.float32)
    for part in VOICE.split(','):
        name, w = part.split(':')
        mix += float(w) * voices[names.index(name)]
    return mix


def load_pronunciations():
    path = os.environ.get('SVA_PRONOUNCE', os.path.join(HERE, 'pronounce.json'))
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}


def split_names(text, pron):
    """['Follow ', ('SVA',), ' Consult.'] -- names become tuples."""
    if not pron:
        return [text]
    pat = re.compile(r"\b(" + '|'.join(re.escape(k) for k in sorted(pron, key=len, reverse=True)) + r")\b['’]?", re.I)
    out, last = [], 0
    for m in pat.finditer(text):
        out.append(text[last:m.start()])
        out.append((m.group(1),))
        last = m.end()
    out.append(text[last:])
    return [p for p in out if p != '']


def trim(x, sr):
    # gentle: soft onsets like the "m" in "Milan" must survive
    loud = np.flatnonzero(np.abs(x) > 0.004)
    if not len(loud):
        return x
    pad = int(0.05 * sr)
    return x[max(0, loud[0] - pad):loud[-1] + pad]


def main():
    out, text = sys.argv[1:3]
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else SPEED
    sym2id, pron = load_tokens(), load_pronunciations()
    pron_l = {k.lower(): v for k, v in pron.items()}
    text = text.replace('...', '…')
    # split after sentence punctuation, keeping it for intonation
    parts = [p.strip() for p in re.split(r'(?<=[.!?…;])\s+', text) if p.strip()]
    spoken = [p.rstrip('…').rstrip() + ('.' if p.endswith('…') else '') for p in parts]
    pieces = [split_names(s, pron) for s in spoken]
    plain = [x.strip() for ps in pieces for x in ps if isinstance(x, str) and re.search(r'\w', x)]
    ph = dict(zip(plain, phonemize(plain))) if plain else {}

    sess = ort.InferenceSession(f'{MODEL}/model.onnx', providers=['CPUExecutionProvider'])
    voice = blended_voice()
    space = sym2id[' ']
    chunks, pos, cues, words = [], 0, [], []
    for p, ps in zip(parts, pieces):
        ids = []
        for x in ps:
            if isinstance(x, tuple):
                seq = [sym2id[c] for c in pron_l[x[0].lower()] if c in sym2id]
            elif re.search(r'\w', x):
                seq = ph[x.strip()]
            else:   # bare punctuation between names, e.g. ", "
                seq = [sym2id[c] for c in x.strip() if c in sym2id]
            if ids and seq and ids[-1] != space and seq[0] != space:
                ids.append(space)
            ids += seq
        style = voice[min(len(ids), FRAMES - 1)].reshape(1, DIM)
        # a short lead-in (space + "…") lets the model fully articulate the first sound
        lead = [sym2id[c] for c in LEAD_IN]
        ids = lead + ids
        audio = sess.run(None, {'tokens': np.array([[0] + ids + [0]], dtype=np.int64),
                                'style': style, 'speed': np.array([speed], dtype=np.float32)})[0]
        x = trim(np.asarray(audio, dtype=np.float32).reshape(-1), SR)
        start, d = pos / SR, len(x) / SR
        cues.append(round(start, 3))
        # word timings are estimated by spreading the sentence over its letters
        ws = p.split()
        weights = [len(w) + 2 for w in ws]
        acc = 0
        for w, k in zip(ws, weights):
            words.append({'w': w, 't': round(start + d * acc / sum(weights), 3)})
            acc += k
        gap = np.zeros(int(PAUSE.get(p[-1], 0.1) * SR), dtype=np.float32)
        chunks += [x, gap]
        pos += len(x) + len(gap)
    samples = np.concatenate(chunks[:-1]) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(out, samples, SR)
    print(json.dumps({'dur': round(len(samples) / SR, 3), 'cues': cues, 'words': words}))


if __name__ == '__main__':
    main()
