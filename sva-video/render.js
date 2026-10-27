// Renders a pick video to MP4 with voiceover and sound design.
//   node render.js picks/pao-milan.json                   -> output/pao-milan.mp4
//   node render.js picks/pao-milan.json --stills 2,5,10   (PNG previews, uses the voice timing)
//   node render.js picks/pao-milan.json --remix           (rebuild audio only, keeps rendered frames)
// A pick can set "template": "template-classic" for the older long-form layout.
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;
// broadcast-style voice chain: clean lows, warmth, presence, tame sibilance,
// gentle compression and a touch of room so it doesn't sound "in a box"
const VOICE_FX = ['highpass=f=75', 'equalizer=f=170:t=q:w=1:g=2', 'equalizer=f=3200:t=q:w=1.2:g=2.5',
  'equalizer=f=7500:t=q:w=1.5:g=-2', 'acompressor=threshold=0.1:ratio=3:attack=5:release=120:makeup=1.5',
  'aecho=0.85:0.5:28|47:0.07|0.04'].join(',');

function tts(text, out) {
  const res = execFileSync('python3', [path.join(__dirname, 'tts.py'), out, text], { stdio: ['ignore', 'pipe', 'ignore'] });
  return JSON.parse(res.toString().trim().split('\n').pop());
}

// Generates every voice clip, stretches each scene to fit its line and hands
// the template sentence cues + word timings (relative to the scene start).
function voiceover(data, workDir, classic) {
  const lead = classic ? 0.4 : 0.15, tail = classic ? 0.6 : 0.25;
  const clips = [];
  fs.mkdirSync(workDir, { recursive: true });
  const say = (key, text) => {
    const file = path.join(workDir, key + '.wav');
    const r = tts(text, file);
    console.log(`  voice ${key}: ${r.dur.toFixed(2)}s`);
    return { file, ...r };
  };
  if (data.voiceIntro) clips.push({ ...say('intro', data.voiceIntro.en), scene: 'intro' });
  data.scenes.forEach((s, i) => {
    if (!s.voice) return;
    const c = say('scene' + i, s.voice.en || s.voice);
    const l = s.lead ?? lead;
    s.len = Math.max(s.dur, l + c.dur + (s.tail ?? tail));
    s.cues = c.cues.map(x => +(x + l).toFixed(3));
    s.words = c.words.map(w => ({ ...w, t: +(w.t + l).toFixed(3) }));
    clips.push({ ...c, scene: i, lead: l });
  });
  if (data.voiceOutro) {
    const c = say('outro', data.voiceOutro.en);
    data.outroLen = c.dur + 1.4;
    clips.push({ ...c, scene: 'outro' });
  }
  return clips;
}

function mix(video, out, dur, clips, tl, data, workDir, classic) {
  const start = c => c.scene === 'intro' ? 0.6 : c.scene === 'outro' ? tl.outro + 0.6 : tl.scenes[c.scene] + c.lead;
  const inputs = [video], filters = [];
  const add = (file, at, vol, label) => {
    inputs.push(file);
    filters.push(`[${inputs.length - 1}]aresample=48000,aformat=channel_layouts=mono,volume=${vol},adelay=${Math.max(0, Math.round(at * 1000))}[${label}]`);
    return `[${label}]`;
  };
  const voice = clips.map((c, i) => add(c.file, start(c), 1, 'v' + i)).join('');
  filters.push(`${voice}amix=inputs=${clips.length}:normalize=0,${VOICE_FX},apad,atrim=0:${dur.toFixed(3)}[vo]`);
  let final = '[vo]';
  if (!classic) {
    const sfxDir = path.join(workDir, 'sfx');
    execFileSync('python3', [path.join(__dirname, 'sfx.py'), sfxDir, String(dur)]);
    const fx = [];
    tl.cuts.forEach((t, i) => fx.push(add(`${sfxDir}/whoosh.wav`, t - 0.34, 0.35, 'w' + i)));
    tl.hits.forEach((t, i) => fx.push(add(`${sfxDir}/impact.wav`, t - 0.02, 0.45, 'h' + i)));
    const pi = data.scenes.findIndex(s => s.type === 'pick');
    if (pi >= 0) {
      const reveal = tl.scenes[pi] + ((data.scenes[pi].cues || [])[1] ?? 1);
      fx.push(add(`${sfxDir}/riser.wav`, reveal - 1.6, 0.22, 'r0'));
    }
    const bed = add(`${sfxDir}/bed.wav`, 0, 0.3, 'bed');
    filters.push(`[vo]asplit=3[vo1][vo2][vo3]`);
    // music ducks under the voice
    filters.push(`${bed}[vo2]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=350[bedd]`);
    filters.push(`${fx.join('')}amix=inputs=${fx.length}:normalize=0[fxraw]`);
    // effects also duck under the voice so they never mask a word
    filters.push(`[fxraw][vo3]sidechaincompress=threshold=0.03:ratio=4:attack=5:release=200[fx]`);
    filters.push(`[vo1][fx][bedd]amix=inputs=3:normalize=0,alimiter=limit=0.9[all]`);
    final = '[all]';
  }
  filters.push(`${final}atrim=0:${dur.toFixed(3)},loudnorm=I=-14:TP=-1.5,aformat=channel_layouts=stereo[a]`);
  const args = ['-y', '-v', 'error'];
  inputs.forEach(f => args.push('-i', f));
  args.push('-filter_complex', filters.join(';'), '-map', '0:v', '-map', '[a]',
    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', out);
  execFileSync('ffmpeg', args, { stdio: 'inherit' });
}

(async () => {
  const args = process.argv.slice(2).filter(a => a !== 'en');
  const [pickFile, flag, stillsArg] = args;
  if (!pickFile) { console.error('usage: node render.js <pick.json> [--stills t1,t2]'); process.exit(1); }
  const data = JSON.parse(fs.readFileSync(pickFile, 'utf8'));
  const classic = data.template === 'template-classic';
  const name = path.basename(pickFile, '.json');
  const outDir = path.join(__dirname, 'output');
  const workDir = path.join(outDir, '.work', name);
  fs.mkdirSync(outDir, { recursive: true });

  const stills = flag === '--stills';
  const hasVoice = (data.scenes || []).some(s => s.voice);
  const clips = hasVoice && !(stills && classic) ? voiceover(data, workDir, classic) : [];

  const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(__dirname, (data.template || 'template') + '.html'));
  const dur = await page.evaluate(d => window.setup(d, 'en'), data);
  const tl = await page.evaluate(() => window.timeline());
  const canvas = await page.$('canvas');

  if (stills) {
    console.log('scenes start at', tl.scenes.map(x => x.toFixed(2)).join(', '), '| total', dur.toFixed(2));
    for (const t of stillsArg.split(',').map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await canvas.screenshot({ path: path.join(outDir, `${name}_t${t}.png`) });
    }
    await browser.close();
    return;
  }

  const out = path.join(outDir, name + '.mp4');
  const silent = clips.length ? path.join(workDir, 'video.mp4') : out;
  // --remix: reuse the rendered frames and only rebuild the audio
  if (flag === '--remix' && fs.existsSync(silent)) {
    await browser.close();
    mix(silent, out, dur, clips, tl, data, workDir, classic);
    console.log('remixed', out);
    return;
  }
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

  if (clips.length) mix(silent, out, dur, clips, tl, data, workDir, classic);
  console.log('wrote', out, `(${dur.toFixed(1)}s)`);
})();
