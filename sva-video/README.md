# SVA Consult – Pick Analysis Videos

Vertical (1080×1920, 30fps, 40s) animated pick-analysis videos in SVA branding.

Scenes: logo intro → matchup → last-5 form → head-to-head stat bars → trend line chart → key insights → THE PICK (odds + confidence) → outro / CTA / 18+.

## Make a new video
1. Copy `picks/aris-aek.json` to e.g. `picks/olympiakos-paok.json` and fill in the real data (English `en` + Greek `gr` text). Set `"sample": false` to remove the "SAMPLE DATA" badge.
2. Render:
   ```
   node render.js picks/olympiakos-paok.json en
   node render.js picks/olympiakos-paok.json gr
   ```
   Videos land in `output/`. Add `--stills 5,12,33` to render quick PNG previews at those seconds instead.

Requires Node, Playwright (Chromium) and ffmpeg.
