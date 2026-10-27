# SVA Consult – Pick Analysis Videos

Vertical (1080×1920, 30fps, 40s) animated pick-analysis videos in SVA branding.

Scenes: logo intro → matchup → last-5 form → head-to-head stat bars → trend line chart → key insights → THE PICK (odds + confidence) → outro / CTA / 18+.

## Make a new video
1. Copy `picks/pao-milan.json` to e.g. `picks/olympiakos-paok.json` and fill in the data.
   Each scene has a `voice.en` line; the scene stretches to fit it so the narration and graphs stay in sync.
   `...` in a voice line adds a dramatic pause.
2. Render:
   ```
   node render.js picks/olympiakos-paok.json en
   ```
   The video lands in `output/`. Add `--stills 5,12,33` to render quick PNG previews instead.

## Voice
Offline Kokoro v1.0 (sherpa-onnx), a blend of `am_puck` and `am_fenrir` with a broadcast EQ/compression chain.
Download `kokoro-multi-lang-v1_0.tar.bz2` from github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models into
`tts-models/` (or point `SVA_TTS_DIR` at it). Change the voice with e.g. `SVA_VOICE=am_michael:1`.

Requires Node, Playwright (Chromium), ffmpeg, and `pip install "sherpa-onnx<2" soundfile`.
