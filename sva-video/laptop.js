// Puts a finished video on a MacBook screen, filmed like a phone shot of a laptop.
//   node laptop.js output/content.mp4 output/content_laptop.mp4 [--pull-start 3.3] [--stills 0,4,10]
// The source video's audio is kept untouched.
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const [src, out] = args;
const pullStart = parseFloat(opt('--pull-start', '0'));
const stills = opt('--stills', null);

// a tiny static server: the page loads the model and frames over http
const server = http.createServer((req, res) => {
  const f = path.join(__dirname, decodeURIComponent(req.url.split('?')[0]));
  if (!f.startsWith(__dirname) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  const type = { '.html': 'text/html', '.js': 'text/javascript', '.glb': 'model/gltf-binary', '.jpg': 'image/jpeg' }[path.extname(f)] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': type });
  fs.createReadStream(f).pipe(res);
});

(async () => {
  await new Promise(r => server.listen(0, r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const work = path.join(__dirname, 'output', '.work', 'laptop-' + path.basename(src, '.mp4'));
  const frames = path.join(work, 'frames');
  fs.mkdirSync(frames, { recursive: true });
  const dur = parseFloat(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src]).toString());
  if (!fs.existsSync(path.join(frames, '00001.jpg'))) {
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', src, '-vf', `fps=${FPS},scale=720:1280`, '-q:v', '3', path.join(frames, '%05d.jpg')]);
  }
  const nFrames = fs.readdirSync(frames).length;

  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto(base + '/laptop.html');
  await page.evaluate(o => window.setup(o), { pullStart, pullDur: 1.4 });
  const canvas = await page.$('canvas');
  const frameUrl = t => `${base}/output/.work/${path.basename(work)}/frames/${String(Math.min(nFrames, Math.floor(t * FPS) + 1)).padStart(5, '0')}.jpg`;

  if (stills) {
    for (const t of stills.split(',').map(Number)) {
      await page.evaluate(([t, u]) => window.render(t, u), [t, frameUrl(t)]);
      await canvas.screenshot({ path: path.join(__dirname, 'output', `laptop_t${t}.png`) });
    }
  } else {
    const silent = path.join(work, 'video.mp4');
    // phone-camera finish: a touch of grain and vignette
    const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
      '-vf', 'noise=alls=5:allf=t,vignette=PI/5', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', silent],
      { stdio: ['pipe', 'inherit', 'inherit'] });
    const total = Math.round(dur * FPS);
    for (let f = 0; f < total; f++) {
      const t = f / FPS;
      await page.evaluate(([t, u]) => window.render(t, u), [t, frameUrl(t)]);
      const buf = await canvas.screenshot({ type: 'jpeg', quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (f % 150 === 0) console.log(`laptop: ${f}/${total}`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', silent, '-i', src, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'copy', '-shortest', '-movflags', '+faststart', out]);
    console.log('wrote', out);
  }
  await browser.close();
  server.close();
})();
