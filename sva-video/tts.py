"""Offline text-to-speech for the voiceover (sherpa-onnx).

    python3 tts.py <lang en|gr> <out.wav> "text to speak" [speed]

Models are expected in $SVA_TTS_DIR (default: ./tts-models):
  en: kokoro-en-v0_19                   (github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
  gr: vits-piper-el_GR-rapunzelina-low
"""
import os
import sys

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = os.environ.get('SVA_TTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts-models'))
EN_SPEAKER = int(os.environ.get('SVA_EN_SPEAKER', '6'))  # kokoro am_michael
SPEED = {'en': 1.12, 'gr': 1.08}


def make(lang):
    if lang == 'en':
        d = os.path.join(ROOT, 'kokoro-en-v0_19')
        model = sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f'{d}/model.onnx', voices=f'{d}/voices.bin', tokens=f'{d}/tokens.txt', data_dir=f'{d}/espeak-ng-data',
            lang='en-us'),
            num_threads=4)
    else:
        d = os.path.join(ROOT, 'vits-piper-el_GR-rapunzelina-low')
        model = sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f'{d}/el_GR-rapunzelina-low.onnx', tokens=f'{d}/tokens.txt', data_dir=f'{d}/espeak-ng-data'),
            num_threads=4)
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model))


def main():
    lang, out, text = sys.argv[1:4]
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else SPEED[lang]
    tts = make(lang)
    audio = tts.generate(text, sid=EN_SPEAKER if lang == 'en' else 0, speed=speed)
    samples = np.array(audio.samples, dtype=np.float32)
    # trim leading/trailing silence so lines start exactly on cue
    loud = np.flatnonzero(np.abs(samples) > 0.02)
    if len(loud):
        pad = int(0.03 * audio.sample_rate)
        samples = samples[max(0, loud[0] - pad):loud[-1] + pad]
    sf.write(out, samples, audio.sample_rate)
    print(f'{len(samples) / audio.sample_rate:.2f}')


if __name__ == '__main__':
    main()
