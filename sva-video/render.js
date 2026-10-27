// Renders a pick video to MP4 (with voiceover when the pick has "voice" text).
//   node render.js picks/pao-milan.json en        -> output/pao-milan_en.mp4
//   node render.js picks/pao-milan.json gr
//   node render.js picks/pao-milan.json en --stills 2,5,10   (PNG previews only)
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const VOICE_LEAD = 0.4;  // seconds between a scene starting and its voice line
const VOICE_TAIL = 0.6;  // breathing room after a line before the next scene
const INTRO_VOICE_AT = 0.6;

function tts(lang, text, out) {
  const secs = execFileSync('python3', [path.join(__dirname, 'tts.py'), lang, out, text], { stdio: ['ignore', 'pipe', 'ignore'] });
  return parseFloat(secs.toString().trim().split('\n').pop());
}

// Generates every voice clip and stretches scenes so each line fits its scene.
function voiceover(data, lang, workDir) {
  const clips = [];
  if (!data.scenes || !data.scenes.some(s => s.voice)) return clips;
  fs.mkdirSync(workDir, { recursive: true });
  const say = (key, obj) => {
    const file = path.join(workDir, key + '.wav');
    const len = tts(lang, obj[lang], file);
    console.log(`  voice ${key}: ${len.toFixed(2)}s`);
    return { file, len };
  };
  if (data.voiceIntro) clips.push({ ...say('intro', data.voiceIntro), scene: 'intro' });
  data.scenes.forEach((s, i) => {
    if (!s.voice) return;
    const c = say('scene' + i, s.voice);
    s.len = Math.max(s.dur, VOICE_LEAD + c.len + VOICE_TAIL);
    clips.push({ ...c, scene: i });
  });
  if (data.voiceOutro) {
    const c = say('outro', data.voiceOutro);
    data.outroLen = c.len + 1.4;
    clips.push({ ...c, scene: 'outro' });
  }
  return clips;
}

(async () => {
  const [pickFile, lang = 'en', flag, stillsArg] = process.argv.slice(2);
  if (!pickFile) { console.error('usage: node render.js <pick.json> <en|gr> [--stills t1,t2]'); process.exit(1); }
  const data = JSON.parse(fs.readFileSync(pickFile, 'utf8'));
  const name = path.basename(pickFile, '.json') + '_' + lang;
  const outDir = path.join(__dirname, 'output');
  const workDir = path.join(outDir, '.work', name);
  fs.mkdirSync(outDir, { recursive: true });

  const stills = flag === '--stills';
  const clips = stills ? [] : voiceover(data, lang, workDir);

  const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(__dirname, 'template.html'));
  const dur = await page.evaluate(([d, l]) => window.setup(d, l), [data, lang]);
  const tl = await page.evaluate(() => window.timeline());
  const canvas = await page.$('canvas');

  if (stills) {
    for (const t of stillsArg.split(',').map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await canvas.screenshot({ path: path.join(outDir, `${name}_t${t}.png`) });
    }
    await browser.close();
    return;
  }

  const out = path.join(outDir, name + '.mp4');
  const silent = clips.length ? path.join(workDir, 'video.mp4') : out;
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', silent],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.round(dur * FPS);
  for (let f = 0; f < total; f++) {
    await page.evaluate(t => window.render(t), f / FPS);
    const buf = await canvas.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 300 === 0) process.stdout.write(`${name}: ${f}/${total}\n`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();

  if (clips.length) {
    // place each line at its scene's start and mix over the video
    const start = c => c.scene === 'intro' ? INTRO_VOICE_AT : c.scene === 'outro' ? tl.outro + 0.6 : tl.scenes[c.scene] + VOICE_LEAD;
    const args = ['-y', '-v', 'error', '-i', silent];
    clips.forEach(c => args.push('-i', c.file));
    const parts = clips.map((c, i) => {
      const ms = Math.round(start(c) * 1000);
      return `[${i + 1}]aresample=48000,aformat=channel_layouts=mono,adelay=${ms}[v${i}]`;
    });
    const mix = clips.map((_, i) => `[v${i}]`).join('') +
      `amix=inputs=${clips.length}:normalize=0,apad,atrim=0:${dur.toFixed(3)},loudnorm=I=-16:TP=-1.5,aformat=channel_layouts=stereo[a]`;
    args.push('-filter_complex', parts.concat(mix).join(';'), '-map', '0:v', '-map', '[a]',
      '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', out);
    execFileSync('ffmpeg', args, { stdio: 'inherit' });
  }
  console.log('wrote', out, `(${dur.toFixed(1)}s)`);
})();
