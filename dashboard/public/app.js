const $ = id => document.getElementById(id);
const prompt = $('prompt');
const runButton = $('run-button');
const runLabel = $('run-label');
const charCount = $('char-count');
const emptyState = $('empty-state');
const resultState = $('result-state');
const outputText = $('output-text');
const detail = $('run-detail');
const copyButton = $('copy-button');
const progressFill = $('progress-fill');
const maxNewInput = $('max-new');
const bosInput = $('bos');
let running = false;
let cancelling = false;
let output = '';
let totalPrompt = 0;
let prefillDone = 0;
let generated = 0;
let currentPrompt = '';
let history = readHistory();

function readHistory() {
  try { return JSON.parse(localStorage.getItem('smollm-history') || '[]').slice(0, 30); }
  catch { return []; }
}
function saveHistory() { localStorage.setItem('smollm-history', JSON.stringify(history.slice(0, 30))); }
function setting(name, fallback) {
  try { const value = localStorage.getItem('smollm-' + name); return value === null ? fallback : value; }
  catch { return fallback; }
}
maxNewInput.value = setting('max-new', '4');
bosInput.checked = setting('bos', 'false') === 'true';
maxNewInput.addEventListener('change', () => {
  const n = Math.max(1, Math.min(64, Number(maxNewInput.value) || 4));
  maxNewInput.value = String(n);
  localStorage.setItem('smollm-max-new', String(n));
});
bosInput.addEventListener('change', () => localStorage.setItem('smollm-bos', String(bosInput.checked)));

function setStatus(label, style = '') {
  $('status-text').textContent = label;
  $('status-pill').className = 'status-pill' + (style ? ` ${style}` : '');
}
function updateControls() {
  charCount.textContent = `${prompt.value.length} ${prompt.value.length === 1 ? 'character' : 'characters'}`;
  runButton.disabled = !running && !prompt.value.trim();
  runButton.classList.toggle('running', running);
  runLabel.textContent = running ? 'Stop' : 'Run';
  $('clear-button').disabled = running;
}
prompt.addEventListener('input', updateControls);
$('clear-button').addEventListener('click', () => { prompt.value = ''; prompt.focus(); updateControls(); });
function showResult() { emptyState.hidden = true; resultState.hidden = false; }
function resetResult() {
  output = ''; totalPrompt = 0; prefillDone = 0; generated = 0;
  outputText.textContent = ''; progressFill.style.width = '0%';
  copyButton.hidden = true; showResult();
  detail.textContent = 'Starting the Verilog engine…';
}
function updateProgress() {
  const maxNew = Number(maxNewInput.value) || 4;
  const total = Math.max(1, totalPrompt + maxNew);
  progressFill.style.width = `${Math.min(100, Math.round(((prefillDone + generated) / total) * 100))}%`;
}
function hexBytes(hex) {
  const bytes = new Uint8Array(Math.floor(hex.length / 2));
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}
function addHistory() {
  history.unshift({prompt:currentPrompt, output, at:Date.now(), maxNew:Number(maxNewInput.value)});
  history = history.slice(0, 30); saveHistory(); renderHistory();
}
function onEvent(message, decoder) {
  switch (message.type) {
    case 'started': setStatus('Running', 'running'); break;
    case 'prompt': totalPrompt = message.total; detail.textContent = `${totalPrompt} prompt ${totalPrompt === 1 ? 'token' : 'tokens'} · preparing model`; break;
    case 'loading': detail.textContent = `Loading model weights · ${totalPrompt} prompt ${totalPrompt === 1 ? 'token' : 'tokens'}`; break;
    case 'processing': detail.textContent = `Processing prompt token ${message.index} of ${message.total}`; break;
    case 'prefill':
      prefillDone = message.done;
      detail.textContent = `Processed ${message.done} of ${message.total} prompt tokens`;
      updateProgress(); break;
    case 'token':
      generated++;
      output += decoder.decode(hexBytes(message.hex), {stream:true});
      outputText.textContent = output;
      detail.textContent = `${generated} ${generated === 1 ? 'token' : 'tokens'} generated`;
      updateProgress(); break;
    case 'error': throw new Error(message.message || 'Inference failed.');
  }
}
async function generate() {
  if (running) {
    cancelling = true;
    detail.textContent = 'Stopping the Verilog engine…';
    await fetch('/api/cancel', {method:'POST'}).catch(() => {});
    return;
  }
  if (!prompt.value.trim()) return;
  currentPrompt = prompt.value;
  const maxNew = Math.max(1, Math.min(64, Number(maxNewInput.value) || 4));
  running = true; cancelling = false; resetResult(); updateControls(); setStatus('Running', 'running');
  const decoder = new TextDecoder();
  let complete = false;
  try {
    const response = await fetch('/api/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:currentPrompt, maxNew, bos:bosInput.checked})
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    const reader = response.body.getReader();
    const textDecoder = new TextDecoder();
    let pending = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += textDecoder.decode(value, {stream:true});
      let at;
      while ((at = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, at);
        pending = pending.slice(at + 1);
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.type === 'done') complete = true;
        else onEvent(message, decoder);
      }
    }
    output += decoder.decode();
    outputText.textContent = output || 'The model ended without visible text.';
    if (cancelling) { setStatus('Stopped'); detail.textContent = 'Generation stopped.'; }
    else if (complete) {
      setStatus('Done', 'done');
      detail.textContent = `${totalPrompt} prompt ${totalPrompt === 1 ? 'token' : 'tokens'} · ${generated} generated`;
      progressFill.style.width = '100%';
      if (output) { copyButton.hidden = false; addHistory(); }
    } else throw new Error('The model stopped before completing.');
  } catch (error) {
    setStatus(cancelling ? 'Stopped' : 'Error', cancelling ? '' : 'error');
    detail.textContent = cancelling ? 'Generation stopped.' : error.message;
    if (!output) outputText.textContent = cancelling ? 'No output generated.' : 'The run did not complete.';
  } finally {
    running = false; cancelling = false; updateControls();
  }
}
runButton.addEventListener('click', generate);
copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(output); copyButton.textContent = 'Copied'; }
  catch { copyButton.textContent = 'Copy failed'; }
  setTimeout(() => { copyButton.textContent = 'Copy output'; }, 1500);
});
function renderHistory() {
  const list = $('history-list'); list.replaceChildren();
  if (!history.length) {
    const empty = document.createElement('div'); empty.className = 'history-empty';
    empty.textContent = 'No runs yet. Generated outputs will appear here.'; list.append(empty); return;
  }
  for (const item of history) {
    const row = document.createElement('button'); row.className = 'history-item'; row.type = 'button';
    const left = document.createElement('span');
    const title = document.createElement('strong'); title.textContent = item.prompt;
    const date = document.createElement('small'); date.textContent = new Date(item.at).toLocaleString();
    left.append(title, date);
    const right = document.createElement('span'); right.className = 'history-output'; right.textContent = item.output || 'No visible output';
    row.append(left, right);
    row.addEventListener('click', () => {
      prompt.value = item.prompt; output = item.output; outputText.textContent = output;
      detail.textContent = `Saved run · ${new Date(item.at).toLocaleString()}`;
      copyButton.hidden = !output; showResult(); setStatus('Saved', 'done');
      progressFill.style.width = '100%'; updateControls(); location.hash = '#workspace';
    });
    list.append(row);
  }
}
$('clear-history').addEventListener('click', () => { history = []; saveHistory(); renderHistory(); });
function updateView() {
  const view = ['workspace','history','settings'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'workspace';
  for (const name of ['workspace','history','settings']) {
    $(`${name}-view`).hidden = name !== view;
    const link = document.querySelector(`[data-view-link="${name}"]`);
    link.classList.toggle('active', name === view);
    if (name === view) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
}
window.addEventListener('hashchange', updateView);
async function checkModel() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    $('model-state').textContent = status.ready ? 'Ready' : 'Unavailable';
    if (!status.ready) { setStatus('Unavailable', 'error'); detail.textContent = 'Model or binary missing.'; }
  } catch { $('model-state').textContent = 'Server offline'; setStatus('Offline', 'error'); }
}
renderHistory(); updateView(); updateControls(); checkModel();
