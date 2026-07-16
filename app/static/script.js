let documentId = null;
let embeddedModels = [];
let selectedEmbedModels = new Set();
let generationModels = [];
let lastResults = [];
let lastQuestions = [];
let lastChunkCount = 0;
let activeTab = "file";

/* ============ helpers ============ */
function setStatus(el, message, kind) {
  el.innerHTML = message;
  el.className = `status-line${kind ? " status-" + kind : ""}`;
}

function setButtonBusy(button, busy, busyLabel, idleLabel) {
  button.disabled = busy;
  button.innerHTML = busy ? `<span class="spinner on-btn"></span>${busyLabel}` : idleLabel;
}

/* An educational "what's happening under the hood" ticker. It does NOT track real
   per-item progress (the backend call is one blocking request) - it walks through
   the conceptual pipeline stages so you can see what the server is doing. */
function startLoadingTicker(box, title, steps, stepMs = 1400) {
  box.classList.remove("hidden");
  box.innerHTML =
    `<div class="loading-title"><span class="spinner"></span>${title}</div>` +
    `<ul class="loading-steps">${steps
      .map((s, i) => `<li data-i="${i}"><span class="dot"></span>${s}</li>`)
      .join("")}</ul>`;
  const items = box.querySelectorAll(".loading-steps li");
  let i = 0;
  const mark = () => {
    items.forEach((li, idx) => {
      li.classList.toggle("active", idx === i);
      li.classList.toggle("done", idx < i);
    });
  };
  mark();
  const timer = setInterval(() => {
    if (i < items.length - 1) {
      i++;
      mark();
    }
  }, stepMs);
  return () => {
    clearInterval(timer);
    box.classList.add("hidden");
    box.innerHTML = "";
  };
}

/* ============ theme ============ */
function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  updateThemeIcon();
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateThemeIcon();
  });
}
function updateThemeIcon() {
  const current =
    document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.getElementById("theme-toggle").textContent = current === "dark" ? "🌙" : "☀️";
}

/* ============ models + ollama panel ============ */
const MAX_EMBED_MODELS = 3;
const modelInfoCache = {}; // model name -> /api/model-info result, fetched lazily on demand

async function loadModels() {
  let data;
  try {
    data = await (await fetch("/api/models")).json();
  } catch {
    data = { ollama_running: false, ollama_installed: false, embedding_models: [], generation_models: [] };
  }

  const banner = document.getElementById("ollama-banner");
  const chipRow = document.getElementById("embedding-model-chips");
  const processBtn = document.getElementById("process-btn");

  renderOllamaPanel(data);

  if (!data.ollama_running) {
    banner.classList.remove("hidden");
    banner.innerHTML = data.ollama_installed
      ? "⚠️ <span>Ollama is installed but not running. Start it (run <code>ollama serve</code> or open the Ollama app), then refresh.</span>"
      : "⚠️ <span>Ollama isn't installed. Get it from <strong>ollama.com</strong>, pull an embedding model and a chat model, then refresh.</span>";
    chipRow.textContent = "Waiting for Ollama…";
    processBtn.disabled = true;
    return;
  }
  banner.classList.add("hidden");

  generationModels = data.generation_models;
  selectedEmbedModels.clear();
  chipRow.innerHTML = "";
  data.embedding_models.forEach((model, idx) => {
    const chip = document.createElement("div");
    chip.className = "model-chip";
    chip.innerHTML = `<span class="check">✓</span>${model}`;
    chip.addEventListener("click", () => toggleEmbedChip(model, chip));
    chipRow.appendChild(chip);
    if (idx === 0) toggleEmbedChip(model, chip); // preselect the first so there's a working default
  });
  if (data.embedding_models.length === 0) {
    chipRow.innerHTML = `No embedding models found. Pull one, e.g. <code>ollama pull nomic-embed-text</code>`;
    processBtn.disabled = true;
  }

  const genOptions = generationModels.map((m) => `<option value="${m}">${m}</option>`).join("");
  document.getElementById("generate-model").innerHTML =
    genOptions || `<option value="">no chat model available</option>`;
  document.getElementById("hyde-model").innerHTML =
    `<option value="">— pick a chat model —</option>` + genOptions;
  document.getElementById("context-model").innerHTML =
    `<option value="">— pick a chat model —</option>` + genOptions;
}

function toggleEmbedChip(model, chip) {
  if (selectedEmbedModels.has(model)) {
    selectedEmbedModels.delete(model);
    chip.classList.remove("selected");
    renderCompat();
    return;
  }
  if (selectedEmbedModels.size >= MAX_EMBED_MODELS) {
    renderCompat(
      `You can compare up to <strong>${MAX_EMBED_MODELS}</strong> embedding models at once. ` +
      `Every chunk gets embedded once per model, so more models means proportionally more work — ` +
      `and for learning, comparing 2–3 side by side is far clearer than a wall of 20. Deselect one to free a slot.`
    );
    return;
  }
  selectedEmbedModels.add(model);
  chip.classList.add("selected");
  renderCompat(); // show "loading limits…" immediately
  ensureModelInfo(model).then(() => renderCompat()); // then fill in the real compatibility note
}

async function ensureModelInfo(model) {
  if (modelInfoCache[model]) return modelInfoCache[model];
  try {
    const info = await (await fetch(`/api/model-info?name=${encodeURIComponent(model)}`)).json();
    modelInfoCache[model] = info;
    return info;
  } catch {
    return null;
  }
}

function explainCompatibility(model, info, chunkSize) {
  const ctx = info.context_length;
  const dim = info.embedding_length;
  const dimTxt = dim ? `, output vector <code>${dim}</code> dims` : "";
  if (!ctx) {
    return `<div class="compat ok"><span class="compat-model">${model}</span> — this model doesn't report an input limit${dimTxt}. Your chunk size <code>${chunkSize}</code> is likely fine, but I can't verify the cap for you.</div>`;
  }
  if (chunkSize > ctx) {
    const dropped = chunkSize - ctx;
    return `<div class="compat warn"><span class="compat-model">${model}</span> — input limit <code>${ctx}</code> tokens${dimTxt}. Your chunk size <code>${chunkSize}</code> is <code>${dropped}</code> tokens over that. ⚠️ Each chunk will be <strong>silently truncated</strong> to the first ${ctx} tokens before embedding, so about ${dropped} tokens per chunk are dropped and never appear in the vector — the model can't retrieve what it never saw. Lower chunk size to ≤ ${ctx} to embed the whole chunk.</div>`;
  }
  const pct = Math.round((chunkSize / ctx) * 100);
  return `<div class="compat ok"><span class="compat-model">${model}</span> — input limit <code>${ctx}</code> tokens${dimTxt}. Your chunk size <code>${chunkSize}</code> fits comfortably (~${pct}% of the limit), so every chunk embeds in full. ✓</div>`;
}

function renderCompat(capMsg) {
  const box = document.getElementById("compat-notes");
  const chunkSize = parseInt(document.getElementById("chunk-size").value, 10) || 0;
  let html = "";
  if (capMsg) html += `<div class="compat cap">${capMsg}</div>`;
  for (const model of selectedEmbedModels) {
    const info = modelInfoCache[model];
    html += info
      ? explainCompatibility(model, info, chunkSize)
      : `<div class="compat loading">Loading ${model}'s limits…</div>`;
  }
  box.innerHTML = html;
}

const META_EXPLANATIONS = {
  capabilities: "What this model can do. 'embedding' = turns text into vectors; 'completion' = generates text.",
  family: "The model's underlying architecture family.",
  parameter_size: "How many learned weights the model has. Bigger usually means smarter but slower and heavier.",
  quantization_level: "How compressed the weights are (e.g. F16, Q4). More compression = smaller & faster, slightly less precise.",
  context_length: "Max tokens it can take as INPUT at once. Go over this and input gets truncated or rejected.",
  embedding_length: "Size of the OUTPUT vector for each text (embedding models only). Fixed no matter how long the input is.",
};

function renderOllamaPanel(data) {
  const status = document.getElementById("panel-status");
  if (data.ollama_running) {
    status.className = "panel-status ok";
    status.textContent = "✓ Ollama is running";
  } else {
    status.className = "panel-status bad";
    status.textContent = data.ollama_installed ? "Ollama installed but not running" : "Ollama not detected";
  }

  const select = document.getElementById("panel-model-select");
  let html = `<option value="">— choose a model —</option>`;
  if (data.embedding_models?.length) {
    html += `<optgroup label="Embedding models (text → vectors)">` +
      data.embedding_models.map((m) => `<option value="${m}">${m}</option>`).join("") + `</optgroup>`;
  }
  if (data.generation_models?.length) {
    html += `<optgroup label="Chat / generation models (write text)">` +
      data.generation_models.map((m) => `<option value="${m}">${m}</option>`).join("") + `</optgroup>`;
  }
  select.innerHTML = html;
  select.onchange = () => { if (select.value) showModelDetail(select.value); };
}

async function showModelDetail(model) {
  const detail = document.getElementById("panel-model-detail");
  detail.innerHTML = `<span class="spinner"></span> Loading ${model}…`;

  const info = await ensureModelInfo(model);
  if (!info) {
    detail.innerHTML = `<span class="status-error">Couldn't load metadata for ${model}.</span>`;
    return;
  }
  const rows = [
    ["capabilities", (info.capabilities || []).join(", ") || "—"],
    ["parameter_size", info.parameter_size],
    ["context_length", info.context_length],
    ["embedding_length", info.embedding_length],
    ["quantization_level", info.quantization_level],
    ["family", info.family],
  ];
  detail.innerHTML = rows
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(
      ([k, v]) => `<div class="meta-row">
        <div><span class="meta-key">${k}</span> &nbsp; <span class="meta-val">${v}</span></div>
        <div class="meta-explain">${META_EXPLANATIONS[k] || ""}</div>
      </div>`
    )
    .join("");
}

function initPanel() {
  const panel = document.getElementById("ollama-panel");
  const overlay = document.getElementById("panel-overlay");
  const open = () => { panel.classList.add("open"); overlay.classList.remove("hidden"); };
  const close = () => { panel.classList.remove("open"); overlay.classList.add("hidden"); };
  document.getElementById("ollama-toggle").addEventListener("click", open);
  document.getElementById("panel-close").addEventListener("click", close);
  overlay.addEventListener("click", close);
}

/* ============ tabs + dropzone ============ */
function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.getElementById("tab-file").classList.toggle("hidden", activeTab !== "file");
      document.getElementById("tab-paste").classList.toggle("hidden", activeTab !== "paste");
    });
  });
}

function initDropzone() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  const chosen = document.getElementById("file-chosen");

  const showChosen = (file) => {
    if (!file) return;
    chosen.classList.remove("hidden");
    chosen.innerHTML = `📎 <strong>${file.name}</strong> <span class="hint">(${(file.size / 1024).toFixed(1)} KB)</span>`;
  };

  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => showChosen(input.files[0]));
  ["dragover", "dragenter"].forEach((e) =>
    dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((e) =>
    dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("dragover"); })
  );
  dz.addEventListener("drop", (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file) { input.files = ev.dataTransfer.files; showChosen(file); }
  });
}

function initContextualToggle() {
  const toggle = document.getElementById("contextual-toggle");
  toggle.addEventListener("change", () => {
    document.getElementById("context-model-wrap").classList.toggle("hidden", !toggle.checked);
  });
}

/* ============ questions ============ */
function addQuestionRow() {
  const container = document.getElementById("question-rows");
  const row = document.createElement("div");
  row.className = "question-row";
  row.innerHTML = `
    <input type="text" class="question-text" placeholder="Ask a question about the document" />
    <input type="text" class="question-keyword" placeholder="Expected keyword(s), comma-separated — optional" />
    <button type="button" class="remove-question icon-btn">✕</button>`;
  row.querySelector(".remove-question").addEventListener("click", () => row.remove());
  container.appendChild(row);
}

function gatherQuestions() {
  const questions = [];
  document.querySelectorAll(".question-row").forEach((row) => {
    const question = row.querySelector(".question-text").value.trim();
    const keyword = row.querySelector(".question-keyword").value.trim();
    if (question) questions.push({ question, expected_keyword: keyword || null });
  });
  return questions;
}

/* ============ step 1: process ============ */
async function handleProcess() {
  const status = document.getElementById("upload-status");
  const button = document.getElementById("process-btn");
  const loadingBox = document.getElementById("upload-loading");

  const models = Array.from(selectedEmbedModels);
  if (models.length === 0) return setStatus(status, "Select at least one embedding model.", "error");

  const formData = new FormData();
  const contextual = document.getElementById("contextual-toggle").checked;
  let contextModel = null;

  if (activeTab === "file") {
    const file = document.getElementById("file-input").files[0];
    if (!file) return setStatus(status, "Choose a file, or switch to the Paste text tab.", "error");
    formData.append("file", file);
  } else {
    const text = document.getElementById("text-input").value.trim();
    if (!text) return setStatus(status, "Paste some text, or switch to the Upload file tab.", "error");
    formData.append("text", text);
  }

  if (contextual) {
    contextModel = document.getElementById("context-model").value;
    if (!contextModel) return setStatus(status, "Contextual Retrieval needs a context-generation model.", "error");
    formData.append("contextual_enabled", "true");
    formData.append("context_model", contextModel);
  }

  const reqInfo = {
    chunkSize: parseInt(document.getElementById("chunk-size").value, 10),
    overlap: parseInt(document.getElementById("overlap").value, 10),
    contextual,
    contextModel,
    source: activeTab === "file" ? "file" : "pasted text",
  };

  formData.append("chunk_size", document.getElementById("chunk-size").value);
  formData.append("overlap", document.getElementById("overlap").value);
  for (const m of models) formData.append("embedding_models", m);

  setStatus(status, "", null);
  const steps = [
    "Reading the document and counting tokens",
    "Splitting text into overlapping chunks",
    contextual ? "Asking a chat model to contextualize each chunk (slow)" : "Preparing chunks for embedding",
    `Embedding every chunk with ${models.length} model(s) — each becomes a vector of numbers`,
    "Storing vectors in memory, ready to search",
  ];
  const stopTicker = startLoadingTicker(loadingBox, "Processing document", steps, contextual ? 2200 : 1200);
  setButtonBusy(button, true, "Processing…", "Process Document");

  try {
    const res = await fetch("/api/documents", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();

    documentId = data.document_id;
    embeddedModels = Object.keys(data.per_model);
    lastChunkCount = data.chunk_count;

    const ctxNote = data.contextual ? " · contextualized" : "";
    let html = `Done — <strong>${data.chunk_count}</strong> chunks, <strong>${data.token_count}</strong> tokens${ctxNote}.`;
    html += `<div class="pill-row">`;
    for (const [m, info] of Object.entries(data.per_model))
      html += `<span class="pill">${m}: dim ${info.dimension}, ${info.embed_time_ms}ms</span>`;
    html += `</div>`;
    setStatus(status, html, "ok");
    showDetails("upload-details-btn", "upload-details", buildProcessDetails(reqInfo, data));

    document.getElementById("section-evaluate").classList.remove("disabled");
    document.getElementById("run-evaluation").disabled = false;
    document.getElementById("section-evaluate").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus(status, `Error: ${err.message}`, "error");
  } finally {
    stopTicker();
    setButtonBusy(button, false, "", "Process Document");
  }
}

/* ============ step 2: evaluate ============ */
async function handleEvaluate() {
  const status = document.getElementById("evaluate-status");
  const button = document.getElementById("run-evaluation");
  const loadingBox = document.getElementById("evaluate-loading");

  const questions = gatherQuestions();
  if (questions.length === 0) return setStatus(status, "Add at least one question.", "error");

  const algorithms = Array.from(document.querySelectorAll(".algorithm-checkbox:checked")).map((c) => c.value);
  if (algorithms.length === 0) return setStatus(status, "Select at least one strategy.", "error");

  const hydeModel = document.getElementById("hyde-model").value;
  if (algorithms.includes("hyde") && !hydeModel)
    return setStatus(status, "HyDE needs a generation model selected.", "error");

  setStatus(status, "", null);
  const combos = embeddedModels.length * algorithms.length;
  const steps = [
    `Embedding your ${questions.length} question(s)`,
    "Searching stored chunks with each strategy",
    algorithms.includes("hyde") ? "HyDE: drafting hypothetical answers first" : "Scoring relevance & redundancy of results",
    `Aggregating metrics across ${combos} combination(s)`,
  ];
  const stopTicker = startLoadingTicker(loadingBox, "Running evaluation", steps, 1300);
  setButtonBusy(button, true, "Running…", "Run Evaluation");

  const body = {
    document_id: documentId,
    questions,
    algorithms,
    embedding_models: embeddedModels,
    generation_model: hydeModel || null,
    k: parseInt(document.getElementById("param-k").value, 10),
    fetch_k: parseInt(document.getElementById("param-fetch-k").value, 10),
    lambda_mult: parseFloat(document.getElementById("param-lambda").value),
  };

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();

    lastResults = data.results;
    lastQuestions = questions.map((q) => q.question);
    setStatus(status, `Done — ${lastResults.length} combination(s) evaluated. Click any row for details.`, "ok");
    renderResultsTable(lastResults);
    showDetails("evaluate-details-btn", "evaluate-details", buildEvaluateDetails(body, lastResults));
    populateGeneratePanel();
  } catch (err) {
    setStatus(status, `Error: ${err.message}`, "error");
  } finally {
    stopTicker();
    setButtonBusy(button, false, "", "Run Evaluation");
  }
}

function metricBar(value) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return `<div class="metric-bar-wrap">${value}<div class="metric-bar"><div class="metric-bar-fill" style="width:${pct}%"></div></div></div>`;
}

function renderResultsTable(results) {
  const wrap = document.getElementById("results-table-wrap");
  const sorted = [...results].sort((a, b) => b.avg_relevance - a.avg_relevance);
  const withHits = sorted.filter((r) => r.hit_rate !== null);
  const bestHit = withHits.length ? Math.max(...withHits.map((r) => r.hit_rate)) : null;

  let html = `<table><thead><tr>
    <th>Embedding Model</th><th>Strategy</th><th>Relevance</th>
    <th>Redundancy</th><th>Hit Rate</th><th>Latency</th>
  </tr></thead><tbody>`;
  sorted.forEach((r, idx) => {
    const isBest = bestHit !== null && r.hit_rate === bestHit && bestHit > 0;
    const hitRate = r.hit_rate === null ? "—" : `<span class="badge${isBest ? " best" : ""}">${Math.round(r.hit_rate * 100)}%</span>`;
    html += `<tr class="result-row" data-idx="${idx}">
      <td>${r.embedding_model}</td><td><span class="badge">${r.algorithm}</span></td>
      <td>${metricBar(r.avg_relevance)}</td><td>${metricBar(r.avg_redundancy)}</td>
      <td>${hitRate}</td><td>${r.avg_latency_ms}ms</td></tr>`;
  });
  html += "</tbody></table>";
  wrap.innerHTML = html;
  wrap.querySelectorAll(".result-row").forEach((row) => {
    row.addEventListener("click", () => toggleDetail(row, sorted[row.dataset.idx]));
  });
}

function interpretRelevance(v) {
  let band;
  if (v >= 0.8) band = `<strong>very high</strong> — the retrieved text closely mirrors the question's meaning`;
  else if (v >= 0.6) band = `<strong>high</strong> — a strong topical match`;
  else if (v >= 0.4) band = `<strong>moderate</strong> — related, but not a tight match; common when a chunk is long and covers many topics (so only part of it answers the question) or when the question is very short`;
  else if (v >= 0.2) band = `<strong>weak</strong> — only loosely related`;
  else band = `<strong>very weak</strong> — likely off-topic`;
  return `<code>${v}</code> is ${band}. This is cosine similarity on a 0→1 scale; absolute values differ by embedding model, so compare rows <em>within the same model</em>, not across different models.`;
}

function interpretRedundancy(v, n) {
  if (n < 2) {
    return `<code>0</code> here means <strong>not enough chunks to measure</strong>, not "perfectly diverse". Redundancy is the average similarity <em>between</em> retrieved chunks — with only ${n} chunk retrieved there's no pair to compare, so it's reported as 0. Try a longer document (more chunks) or a higher k to get a real value.`;
  }
  let band;
  if (v >= 0.7) band = `<strong>high</strong> — the ${n} retrieved chunks are near-duplicates, so your retrieval budget is spent on repeated information`;
  else if (v >= 0.4) band = `<strong>moderate</strong> — some overlap between chunks, but reasonable variety`;
  else band = `<strong>low</strong> — the retrieved chunks are well varied, covering different parts of the document`;
  return `<code>${v}</code> is ${band}. (This is what MMR tries to push down.)`;
}

function interpretHit(q) {
  const searched = (q.hit_keywords || []).map((k) => `<code>${escapeHtml(k)}</code>`).join(", ");
  if (q.hit) {
    const matched = q.hit_matched.map((k) => `<code>${escapeHtml(k)}</code>`).join(", ");
    return `matched ${matched} inside the retrieved text (searched: ${searched}), so the chunk that answers this question was successfully retrieved.`;
  }
  return `none of your keyword(s) (${searched}) appeared in the retrieved text — a MISS. Either retrieval didn't surface the right chunk, or your keyword isn't a literal match for how the document words it.`;
}

function toggleDetail(row, result) {
  const existing = row.nextElementSibling;
  if (existing && existing.classList.contains("detail-row")) return existing.remove();
  const detailRow = document.createElement("tr");
  detailRow.className = "detail-row";
  const cell = document.createElement("td");
  cell.colSpan = 6;
  let html = "";
  for (const q of result.per_question) {
    const n = q.retrieved_chunks.length;
    const hitLabel = q.hit === null ? "" : q.hit
      ? '<span class="badge hit-badge">HIT</span>'
      : '<span class="badge miss-badge">MISS</span>';
    html += `<p><strong>${escapeHtml(q.question)}</strong> ${hitLabel}<br/>
      <span class="hint">relevance=${q.relevance} · redundancy=${q.redundancy} · ${n} chunk(s) retrieved · ${q.latency_ms}ms</span></p>`;
    html += `<div class="metric-explain">
      <div>📊 <strong>Relevance</strong> — ${interpretRelevance(q.relevance)}</div>
      <div>♻️ <strong>Redundancy</strong> — ${interpretRedundancy(q.redundancy, n)}</div>
      ${q.hit !== null ? `<div>🎯 <strong>Hit</strong> — ${interpretHit(q)}</div>` : ""}
    </div>`;
    html += `<div class="hint" style="margin:0.3rem 0 0.2rem">Retrieved chunk(s):</div>`;
    for (const chunk of q.retrieved_chunks) html += `<div class="chunk-preview">${escapeHtml(chunk)}</div>`;
  }
  cell.innerHTML = html;
  detailRow.appendChild(cell);
  row.after(detailRow);
}

/* ============ step 3: generate ============ */
function populateGeneratePanel() {
  document.getElementById("generate-combo").innerHTML = lastResults
    .map((r, i) => `<option value="${i}">${r.embedding_model} / ${r.algorithm}</option>`).join("");
  document.getElementById("generate-question").innerHTML = lastQuestions
    .map((q) => `<option value="${escapeHtml(q)}">${escapeHtml(q)}</option>`).join("");
  document.getElementById("section-generate").classList.remove("disabled");
  document.getElementById("run-generate").disabled = false;
}

async function handleGenerate() {
  const status = document.getElementById("generate-status");
  const button = document.getElementById("run-generate");
  const loadingBox = document.getElementById("generate-loading");
  const resultDiv = document.getElementById("generate-result");

  const combo = lastResults[parseInt(document.getElementById("generate-combo").value, 10)];
  const question = document.getElementById("generate-question").value;
  const generationModel = document.getElementById("generate-model").value;
  if (!generationModel) return setStatus(status, "No chat model available — pull one in Ollama first.", "error");

  setStatus(status, "", null);
  resultDiv.innerHTML = "";
  const steps = [
    "Retrieving the most relevant chunks",
    "Building a prompt: 'answer using ONLY this context'",
    `Asking ${generationModel} to write the answer`,
  ];
  const stopTicker = startLoadingTicker(loadingBox, "Generating answer", steps, 1500);
  setButtonBusy(button, true, "Generating…", "Generate");

  const body = {
    document_id: documentId,
    question,
    embedding_model: combo.embedding_model,
    algorithm: combo.algorithm,
    generation_model: generationModel,
    k: parseInt(document.getElementById("param-k").value, 10),
    fetch_k: parseInt(document.getElementById("param-fetch-k").value, 10),
    lambda_mult: parseFloat(document.getElementById("param-lambda").value),
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    setStatus(status, `Done (${data.latency_ms}ms).`, "ok");
    let html = "<h3>Retrieved context</h3>";
    for (const chunk of data.retrieved_chunks) html += `<div class="chunk-preview">${escapeHtml(chunk)}</div>`;
    html += `<h3>Answer</h3><p>${escapeHtml(data.answer)}</p>`;
    resultDiv.innerHTML = html;
    showDetails("generate-details-btn", "generate-details", buildGenerateDetails(body, data));
  } catch (err) {
    setStatus(status, `Error: ${err.message}`, "error");
  } finally {
    stopTicker();
    setButtonBusy(button, false, "", "Generate");
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/* ============ "how did this work?" details ============ */
function showDetails(btnId, boxId, html) {
  document.getElementById(boxId).innerHTML = html;
  document.getElementById(btnId).classList.remove("hidden");
}

function wireDetailToggles() {
  [
    ["upload-details-btn", "upload-details"],
    ["evaluate-details-btn", "evaluate-details"],
    ["generate-details-btn", "generate-details"],
  ].forEach(([btnId, boxId]) => {
    document.getElementById(btnId).addEventListener("click", () =>
      document.getElementById(boxId).classList.toggle("hidden")
    );
  });
}

function buildProcessDetails(info, data) {
  const step = info.chunkSize - info.overlap;
  const s = [];
  s.push(`<strong>Read &amp; tokenize.</strong> Your ${info.source} was read as plain text and split into tokens with <code>tiktoken</code> (cl100k_base). It came to <code>${data.token_count}</code> tokens — tokens aren't words, they're frequency-based sub-word pieces.`);
  s.push(`<strong>Chunk.</strong> A window of <code>${info.chunkSize}</code> tokens slid across the text, advancing <code>${step}</code> tokens each step (<code>${info.chunkSize} − ${info.overlap}</code> overlap), so neighbors share <code>${info.overlap}</code> tokens. That produced <code>${data.chunk_count}</code> chunks.`);
  if (data.contextual) {
    s.push(`<strong>Contextualize.</strong> Contextual Retrieval was ON, so for each of the ${data.chunk_count} chunks, <code>${info.contextModel}</code> wrote a 1–2 sentence blurb describing what the chunk is about and prepended it before embedding.`);
  }
  for (const [m, i] of Object.entries(data.per_model)) {
    s.push(`<strong>Embed with ${m}.</strong> All ${data.chunk_count} ${data.contextual ? "contextualized " : ""}chunks were sent to Ollama and turned into <code>${i.dimension}</code>-dimensional vectors (${i.embed_time_ms} ms). Every chunk becomes the same ${i.dimension} numbers long, regardless of its length.`);
  }
  s.push(`<strong>Store.</strong> Each vector was saved in memory next to its original chunk text under a document id, ready to search in step 2.`);
  return `<p>Exactly what happened when you clicked Process Document:</p><ol class="detail-steps">${s.map((x) => `<li>${x}</li>`).join("")}</ol>`;
}

function buildEvaluateDetails(body, results) {
  const algoText = {
    topk: `<strong>Top-K</strong> embedded the question, compared it against every stored chunk vector with cosine similarity, and kept the <code>${body.k}</code> closest.`,
    mmr: `<strong>MMR</strong> took the top <code>${body.fetch_k}</code> by similarity, then greedily picked <code>${body.k}</code>, each time balancing relevance against how different a candidate is from what's already chosen (lambda=<code>${body.lambda_mult}</code>).`,
    hybrid: `<strong>Hybrid</strong> built two rankings over the top <code>${body.fetch_k}</code> — meaning-based (vectors) and keyword-based (BM25) — then fused them with Reciprocal Rank Fusion and kept the top <code>${body.k}</code>.`,
    hyde: `<strong>HyDE</strong> first asked the chat model to draft a hypothetical answer, embedded <em>that</em> instead of your question, then ran Top-K (<code>${body.k}</code>).`,
  };
  const s = [];
  s.push(`<strong>Set up the grid.</strong> You compared <code>${body.embedding_models.length}</code> embedding model(s) × <code>${body.algorithms.length}</code> strategy(ies) across <code>${body.questions.length}</code> question(s) = <code>${results.length}</code> combinations.`);
  s.push(`<strong>Per question.</strong> Each question was embedded with the same model that embedded the chunks (so they share one vector space), then retrieved by each strategy:<ul>${body.algorithms.map((a) => `<li>${algoText[a] || a}</li>`).join("")}</ul>`);
  s.push(`<strong>Scored the results.</strong> <em>Relevance</em> = average cosine similarity between the question and each retrieved chunk (higher = more on-topic). <em>Redundancy</em> = average similarity among the retrieved chunks themselves (lower = more varied). <em>Latency</em> = retrieval time.`);
  s.push(`<strong>Checked for hits.</strong> Your "expected keyword" field is split on commas into separate keywords (so <code>Django, FastAPI</code> = two acceptable keywords), each lower-cased and searched as plain text inside the retrieved chunks. It counts as a HIT if <em>any</em> of them appears. Expand a row above to see exactly which keywords were searched and which matched.`);
  s.push(`<strong>Hit rate ≠ answer quality.</strong> Hit rate only asks "did the <em>retrieval</em> step pull a chunk containing your keyword" — a rough proxy. The grounded answer in step 3 can still be perfectly correct even on a MISS, because the model reads whatever was retrieved. If retrieval pulled the right chunk but your keyword wasn't a literal match, you'll see MISS yet a correct answer — that's the metric being crude, not retrieval failing.`);
  if (lastChunkCount && lastChunkCount <= body.k) {
    s.push(`<strong>Heads up: only ${lastChunkCount} chunk(s).</strong> With so few chunks, retrieval returns essentially everything every time, so relevance and hit rate can't really distinguish strategies here. Paste a longer document (many chunks) to see the strategies actually diverge.`);
  }
  const withHits = results.filter((r) => r.hit_rate !== null);
  if (withHits.length) {
    const best = withHits.reduce((a, b) => (b.hit_rate > a.hit_rate ? b : a));
    s.push(`<strong>Read the table.</strong> Best hit rate was <code>${Math.round(best.hit_rate * 100)}%</code> (${best.embedding_model} / ${best.algorithm}). Watch for high relevance but low hit rate — that means "confidently retrieved the wrong thing."`);
  } else {
    s.push(`<strong>Tip.</strong> Add an "expected keyword" to a question to unlock the Hit Rate column — the only true accuracy check here.`);
  }
  return `<p>What "Run Evaluation" actually did:</p><ol class="detail-steps">${s.map((x) => `<li>${x}</li>`).join("")}</ol>`;
}

function buildGenerateDetails(body, data) {
  const promptTemplate =
    `Answer the question using ONLY the context below.\n` +
    `If the answer is not contained in the context, say "I don't know based on the given context."\n\n` +
    `Context:\n{the ${data.retrieved_chunks.length} retrieved chunk(s), joined together}\n\n` +
    `Question: ${body.question}\n\nAnswer:`;
  const s = [];
  s.push(`<strong>Retrieve.</strong> Using <code>${body.embedding_model}</code> + <code>${body.algorithm}</code>, the ${data.retrieved_chunks.length} most relevant chunk(s) were pulled from storage.`);
  s.push(`<strong>Build the prompt.</strong> Those chunks were concatenated and wrapped in a strict instruction, so the model can't fall back on training knowledge:<div class="detail-prompt">${escapeHtml(promptTemplate)}</div>`);
  s.push(`<strong>Generate.</strong> That prompt went to <code>${body.generation_model}</code>, which wrote the answer grounded only in the context (${data.latency_ms} ms). If the answer isn't in the chunks, a well-behaved model says it doesn't know — that's RAG working, not failing.`);
  return `<p>How that answer was produced:</p><ol class="detail-steps">${s.map((x) => `<li>${x}</li>`).join("")}</ol>`;
}

/* ============ boot ============ */
initTheme();
initPanel();
initTabs();
initDropzone();
initContextualToggle();
wireDetailToggles();
document.getElementById("process-btn").addEventListener("click", handleProcess);
document.getElementById("chunk-size").addEventListener("input", () => renderCompat());
document.getElementById("add-question").addEventListener("click", addQuestionRow);
document.getElementById("run-evaluation").addEventListener("click", handleEvaluate);
document.getElementById("run-generate").addEventListener("click", handleGenerate);
addQuestionRow();
loadModels();
