// Renders a pick video to MP4.
//   node render.js picks/aris-aek.json en        -> output/aris-aek_en.mp4
//   node render.js picks/aris-aek.json gr
//   node render.js picks/aris-aek.json en --stills 2,5,10   (PNG previews only)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;

(async () => {
  const [pickFile, lang = 'en', flag, stillsArg] = process.argv.slice(2);
  if (!pickFile) { console.error('usage: node render.js <pick.json> <en|gr> [--stills t1,t2]'); process.exit(1); }
  const data = JSON.parse(fs.readFileSync(pickFile, 'utf8'));
  const name = path.basename(pickFile, '.json') + '_' + lang;
  const outDir = path.join(__dirname, 'output');
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(__dirname, 'template.html'));
  const dur = await page.evaluate(([d, l]) => window.setup(d, l), [data, lang]);
  const canvas = await page.$('canvas');

  if (flag === '--stills') {
    for (const t of stillsArg.split(',').map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await canvas.screenshot({ path: path.join(outDir, `${name}_t${t}.png`) });
    }
    await browser.close();
    return;
  }

  const out = path.join(outDir, name + '.mp4');
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.round(dur * FPS);
  for (let f = 0; f < total; f++) {
    await page.evaluate(t => window.render(t), f / FPS);
    const buf = await canvas.screenshot({ type: 'jpeg', quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 150 === 0) process.stdout.write(`${name}: ${f}/${total}\n`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.log('wrote', out);
})();
