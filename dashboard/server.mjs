import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const engine = resolve(here, '..');
const publicDir = join(here, 'public');
const binary = join(engine, 'obj_dir', 'Vsmollm');
const source = join(engine, 'smollm.sv');
const model = join(engine, 'data', 'model.safetensors');
const host = '127.0.0.1';
const port = Number(process.env.PORT || 4173);
let active = null;

async function exists(path) {
  try { return await stat(path); } catch { return null; }
}

async function buildIfNeeded() {
  const [bin, src, weights] = await Promise.all([exists(binary), exists(source), exists(model)]);
  if (!src || !weights) throw new Error('Verilog source or model checkpoint is missing.');
  if (bin && bin.mtimeMs >= src.mtimeMs) return;
  console.log('Compiling the Verilog engine (about 5 seconds)...');
  const build = spawnSync('verilator', ['--binary', '--timing', '-O3', '-Wno-fatal', '-j', '4', 'smollm.sv'], {
    cwd: engine, stdio: 'inherit', timeout: 180_000
  });
  if (build.error) throw build.error;
  if (build.status !== 0) throw new Error(`Verilator exited with code ${build.status}.`);
}

function json(res, status, value) {
  res.writeHead(status, {'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});
  res.end(JSON.stringify(value));
}

function event(res, value) {
  if (!res.destroyed && !res.writableEnded) res.write(JSON.stringify(value) + '\n');
}

function parseLine(res, line) {
  if (!line.startsWith('EVT ')) return false;
  const rest = line.slice(4);
  if (rest.startsWith('PROMPT ')) event(res, {type:'prompt', total:Number(rest.slice(7))});
  else if (rest === 'LOADING') event(res, {type:'loading'});
  else if (rest.startsWith('PROCESSING ')) {
    const [index,total] = rest.slice(11).split(' ').map(Number);
    event(res, {type:'processing', index, total});
  }
  else if (rest.startsWith('PREFILL ')) {
    const [done,total] = rest.slice(8).split(' ').map(Number);
    event(res, {type:'prefill', done, total});
  } else if (rest.startsWith('TOKEN ')) event(res, {type:'token', hex:rest.slice(6)});
  else if (rest === 'DONE') event(res, {type:'done'});
  return rest === 'DONE';
}

async function run(req, res) {
  if (active) return json(res, 409, {error:'A generation is already running.'});
  const origin = req.headers.origin;
  if (origin && origin !== `http://${host}:${port}` && origin !== `http://localhost:${port}`)
    return json(res, 403, {error:'This dashboard only accepts local requests.'});
  if (!String(req.headers['content-type'] || '').startsWith('application/json'))
    return json(res, 415, {error:'Expected JSON.'});
  let body = '';
  try {
    for await (const chunk of req) {
      body += chunk;
      if (body.length > 32_768) throw new Error('Prompt is too long.');
    }
    const input = JSON.parse(body);
    const prompt = String(input.prompt ?? '');
    const maxNew = Number(input.maxNew);
    const bos = input.bos === true;
    if (!prompt.trim()) throw new Error('Enter a prompt first.');
    if (Buffer.byteLength(prompt, 'utf8') > 8192) throw new Error('Prompt exceeds 8,192 bytes.');
    if (!Number.isInteger(maxNew) || maxNew < 1 || maxNew > 64)
      throw new Error('Max new tokens must be between 1 and 64.');
    const args = [`+prompt=${prompt}`, `+max_new=${maxNew}`, '+machine'];
    if (bos) args.push('+bos');
    const child = spawn(binary, args, {cwd:engine, stdio:['ignore','pipe','pipe']});
    active = {child};
    res.writeHead(200, {
      'Content-Type':'application/x-ndjson; charset=utf-8',
      'Cache-Control':'no-store',
      'X-Content-Type-Options':'nosniff'
    });
    event(res, {type:'started'});
    let pending = '';
    let stderr = '';
    let done = false;
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      pending += chunk;
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline).trimEnd();
        pending = pending.slice(newline + 1);
        if (parseLine(res, line)) done = true;
      }
    });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-2048); });
    child.on('error', error => { event(res, {type:'error', message:error.message}); });
    child.on('close', code => {
      if (active?.child === child) active = null;
      if (code !== 0) event(res, {type:'error', message:stderr.trim() || `Verilog process exited with code ${code}.`});
      else if (!done) event(res, {type:'done'});
      if (!res.destroyed && !res.writableEnded) res.end();
    });
    res.on('close', () => { if (!res.writableEnded && active?.child === child) child.kill('SIGTERM'); });
  } catch (error) {
    if (!res.headersSent) json(res, 400, {error:error.message});
    else { event(res, {type:'error', message:error.message}); res.end(); }
  }
}

const mime = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8',
  '.js':'text/javascript; charset=utf-8','.ttf':'font/ttf','.svg':'image/svg+xml'};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${host}:${port}`);
  if (req.method === 'GET' && url.pathname === '/api/status') {
    const [bin, weights] = await Promise.all([exists(binary), exists(model)]);
    return json(res, 200, {ready:!!bin && !!weights, busy:!!active, context:2048,
      model:'SmolLM-135M', modelBytes:weights?.size ?? 0});
  }
  if (req.method === 'POST' && url.pathname === '/api/run') return run(req, res);
  if (req.method === 'POST' && url.pathname === '/api/cancel') {
    const origin = req.headers.origin;
    if (origin && origin !== `http://${host}:${port}` && origin !== `http://localhost:${port}`)
      return json(res, 403, {error:'This dashboard only accepts local requests.'});
    if (active) active.child.kill('SIGTERM');
    return json(res, 200, {cancelled:!!active});
  }
  if (req.method !== 'GET') return json(res, 405, {error:'Method not allowed.'});
  const name = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname.slice(1));
  const file = resolve(publicDir, name);
  if (relative(publicDir, file).startsWith('..')) return json(res, 403, {error:'Forbidden.'});
  try {
    const content = await readFile(file);
    res.writeHead(200, {'Content-Type':mime[extname(file)] || 'application/octet-stream',
      'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    res.end(content);
  } catch { json(res, 404, {error:'Not found.'}); }
});

try {
  await buildIfNeeded();
  server.on('error', error => { console.error(`Dashboard server: ${error.message}`); process.exitCode = 1; });
  server.listen(port, host, () => console.log(`Prompt Studio: http://${host}:${port}`));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
