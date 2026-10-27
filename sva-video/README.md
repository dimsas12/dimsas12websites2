# SVA Consult – Pick Analysis Videos

Short vertical (1080×1920, 30fps, ~30s) pick videos built to hook in the first second:
cold-open hook → versus → stat duel → totals vs line → head-to-head → model → THE PICK → SVA logo outro.
Brand-slash / whip-pan / zoom / glitch transitions, voice-cued impacts, word-by-word captions,
voiceover and synthesised sound design (whooshes, impacts, riser, ducked beat).

## Make a new video
1. Copy `picks/pao-milan.json` to e.g. `picks/olympiakos-paok.json` and fill it in.
   Every scene has a `voice` line; the scene stretches to fit it and its animations fire on the
   sentence/word cues, so graphics, captions and sound stay in sync. `...` adds a dramatic pause.
   Scene types: `hook`, `versus`, `duel`, `totals`, `numbers`, `model`, `pick`, `outro` (see the example).
2. Render:
   ```
   node render.js picks/olympiakos-paok.json
   ```
   The video lands in `output/`. Add `--stills 1,5,12` to render PNG previews at those seconds instead.

Older long-form picks set `"template": "template-classic"`.

## Voice
Offline Kokoro v1.0 (sherpa-onnx), a blend of `am_puck` and `am_fenrir` with a broadcast EQ/compression chain.
Download `kokoro-multi-lang-v1_0.tar.bz2` from github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models into
`tts-models/` (or point `SVA_TTS_DIR` at it). Change the voice with `SVA_VOICE=am_michael:1`, pace with `SVA_SPEED`.

### Pronunciation
Names are spoken from `pronounce.json` (word → IPA), e.g. `"Panathinaikos": "pˌɑnɑθˌinɑikˈɔs"`.
Add every new team/player name there before rendering; the rest of each sentence is phonemized by espeak
and the model runs on the combined phonemes, so names come out right with no pauses.

Requires Node, Playwright (Chromium), ffmpeg, and `pip install "sherpa-onnx<2" onnxruntime soundfile scipy`.
