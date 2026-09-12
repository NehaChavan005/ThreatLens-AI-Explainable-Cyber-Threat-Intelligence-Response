'use strict';

const el = (id) => document.getElementById(id);
const cfg = {
  base: localStorage.getItem('tl_base') || (location.origin),
  key: localStorage.getItem('tl_key') || '',
};

let charts = { dist: null, timeline: null };

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if (cfg.key) h['X-API-Key'] = cfg.key;
  return h;
}

function setStatus(cls, text) {
  const s = el('conn-status');
  s.className = 'conn-badge conn-' + cls;
  s.textContent = text;
}

async function api(path, opts = {}) {
  const res = await fetch(cfg.base + path, { ...opts, headers: headers() });
  if (res.status === 401 || res.status === 403 || res.status === 422) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status} — ${body.slice(0, 200)}`);
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.status === 204 ? null : res.json();
}

function saveConfig() {
  cfg.base = el('api-base').value.trim().replace(/\/+$/, '');
  cfg.key = el('api-key').value.trim();
  localStorage.setItem('tl_base', cfg.base);
  localStorage.setItem('tl_key', cfg.key);
  setStatus('idle', 'saved');
  boot();
}

/* ------------------------------------------------------- severity helpers */
const SEV_ORDER = { low: 0, medium: 1, high: 2, critical: 3 };
function badge(sev) {
  const s = (sev || 'medium').toLowerCase();
  return `<span class="sev-badge sev-${s}">${sev.toUpperCase()}</span>`;
}
function threatClass(sev) {
  return 'threat-' + ((sev || 'medium').toLowerCase());
}
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

/* ------------------------------------------------------- prediction */
function validJson(txt) {
  try { return JSON.parse(txt); } catch (e) { return null; }
}
function fmtSampleFeatures(obj) {
  return JSON.stringify(obj, null, 1);
}
function dominantFeature(explanation) {
  if (!explanation || !explanation.top_features || !explanation.top_features.length) return '(';
  const a = explanation.top_features[0].shap_value;
  return a && a > 0 ? 'flag' : 'ok';
}

function renderThreat(data) {
  const panel = el('result-panel');
  panel.style.display = 'block';
  const threat = el('result-threat');
  const sev = data.severity || 'medium';

  let banner;
  if (data.is_attack) {
    banner = `
      <div class="threat-meta">Prediction is authoritative; AI explanations & advisory are for validation.</div>
    `;
  }

  threat.className = 'result-threat ' + threatClass(sev);
  threat.innerHTML = data.is_attack ? `
    <span>🚨 THREAT DETECTED</span>
    <span>${esc(data.prediction_class)}</span>
    <span>${badge(sev)}</span>
    <span class="threat-meta">confidence ${(data.confidence * 100).toFixed(1)}%</span>
    <span class="threat-meta">backend ${esc(data.model_backend)}</span>
    ${banner}
  ` : `
    <span>✅ BENIGN FLOW</span>
    <span class="threat-meta">confidence ${(data.confidence * 100).toFixed(1)}%</span>
    <span class="threat-meta">backend ${esc(data.model_backend)}</span>
  `;
  renderExplain(data.explanation);
  renderAdvisory(data.advisory);
  renderActions(data.defensive_actions, data.alert_id);

  if (data.alert_id) {
    el('result-alert-link').innerHTML = `
      <h3>Alert & Incident</h3>
      <p><a href="#incidents" data-jump="true">Alert #${data.alert_id} created</a> — correlated incident below (auto-linked by source IP).</p>
    `;
  } else {
    el('result-alert-link').innerHTML = '';
  }
}

function renderExplain(explanation) {
  const box = el('result-explain');
  const bars = box.querySelector('.shap-bars');
  const evd = box.querySelector('.evidence');
  if (!explanation || explanation.status !== 'ok' || !explanation.top_features || !explanation.top_features.length) {
    bars.innerHTML = '<p class="muted">No SHAP explanation available for this backend.</p>';
    evd.innerHTML = '';
    return;
  }
  const maxAbs = Math.max(...explanation.top_features.map(f => Math.abs(f.shap_value)), 1e-9);
  bars.innerHTML = explanation.top_features.map(f => {
    const cls = f.direction === 'increase' || f.shap_value > 0 ? 'shap-increase' : 'shap-decrease';
    const width = (Math.abs(f.shap_value) / maxAbs * 100).toFixed(1);
    return `<div class="shap-bar-row" title="${esc(f.feature)} = ${esc(f.raw_value)}">
      <span class="shap-feature">${esc(f.feature)}</span>
      <span class="shap-bar-track"><span class="shap-bar-fill ${cls}" style="width:${width}%"></span></span>
      <span class="shap-val">${f.shap_value >= 0 ? '+' : ''}${f.shap_value.toFixed(4)}</span>
    </div>`;
  }).join('');
  evd.innerHTML = explanation.interpretation
    ? `<p class="muted" style="font-size:0.78rem;color:var(--text-dim)">${esc(explanation.interpretation)}</p>`
    : '';
  if (explanation.evidence_summary && explanation.evidence_summary.length) {
    evd.innerHTML += `<ul class="evidence">${explanation.evidence_summary.map(e => `<li>${esc(e)}</li>`).join('')}</ul>`;
  }
}

function renderAdvisory(advisory) {
  el('result-advisory').querySelector('.advisory-text').innerHTML = advisory && advisory.advisory
    ? esc(advisory.advisory)
    : '<p class="muted">No advisory returned.</p>';
}

function renderActions(actions, alertId) {
  const box = el('result-actions');
  const list = box.querySelector('.action-list');
  if (!actions || !actions.length) {
    list.innerHTML = '<li class="muted" style="color:var(--text-dim)">No defensive actions recommended for a benign flow.</li>';
    return;
  }
  list.innerHTML = actions.map(a => `
    <li class="priority-${esc(a.priority)}">
      <div class="action-name">${esc(a.action)}</div>
      <div class="action-reason">${esc(a.reason)}</div>
      <div class="action-meta">priority ${esc(a.priority)} · urgency ${esc(a.urgency)} · ${a.approval_required ? 'analyst approval required' : 'informational'}</div>
    </li>`).join('');
}

async function runPrediction() {
  const txt = el('flow-json').value.trim();
  const body = validJson(txt);
  if (!body) { alert('Feature JSON is not valid JSON.'); return; }
  // PredictionRequest is a FLAT body: canonical feature fields + src/dst + flags.
  const payload = body;
  payload.include_explanation = el('chk-explain').checked;
  payload.include_advisory = el('chk-advisory').checked;
  try {
    const data = await api('/predict', { method: 'POST', body: JSON.stringify(payload) });
    renderThreat(data);
  } catch (e) {
    alert('Prediction failed: ' + e.message);
  }
}

/* ------------------------------------------------------- dashboards */
async function loadSample(which) {
  try {
    const r = await fetch(cfg.base + '/dashboard/' + which + '.json');
    const obj = await r.json();
    el('flow-json').value = fmtSampleFeatures(obj);
    renderThreatPlaceholder();
  } catch (e) {
    alert('Could not load sample: ' + e.message);
  }
}
function renderThreatPlaceholder() {
  el('result-panel').style.display = 'block';
  el('result-threat').className = 'result-threat threat-idle';
  el('result-threat').innerHTML = '<span>Sample flow loaded — click Run Prediction</span>';
}

async function loadStats() {
  try {
    const s = await api('/api/dashboard/stats?hours=24');
    el('stat-total-alerts').querySelector('.stat-value').textContent = s.total_alerts;
    el('stat-total-incidents').querySelector('.stat-value').textContent = s.total_incidents;
    el('stat-active-incidents').querySelector('.stat-value').textContent = s.active_incidents;
    const sev = ['critical', 'high', 'medium'].reduce((m, k) => { m[k] = s.alerts_by_severity[k] || 0; return m; }, {});
    el('stat-critical').querySelector('.stat-value').textContent = sev.critical;
    el('stat-high').querySelector('.stat-value').textContent = sev.high;
    el('stat-medium').querySelector('.stat-value').textContent = sev.medium;
  } catch (e) { /* dashboard is optional */ }
}

async function loadRecentAlerts() {
  try {
    const alerts = await api('/api/dashboard/recent-alerts?limit=20');
    const tbody = el('table-alerts').querySelector('tbody');
    if (!alerts.length) { tbody.innerHTML = '<tr><td colspan="7">No alerts recorded yet.</td></tr>'; return; }
    tbody.innerHTML = alerts.map(a => `
      <tr>
        <td>${a.id}</td>
        <td>${esc(a.attack_type)}</td>
        <td>${badge(a.severity)}</td>
        <td>${esc(a.src_ip)}${a.dst_ip ? ' → ' + esc(a.dst_ip) : ''}</td>
        <td>${a.confidence != null ? (a.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td>${new Date(a.timestamp).toLocaleString()}</td>
        <td>${a.acknowledged ? '✓' : ''}</td>
      </tr>
    `).join('');
  } catch (e) { /* optional */ }
}

async function loadIncidents() {
  try {
    const incidents = await api('/api/alerts/incidents');
    const tbody = el('table-incidents').querySelector('tbody');
    if (!incidents.length) { tbody.innerHTML = '<tr><td colspan="6">No incidents correlation (needs ≥2 alerts from the same source IP).</td></tr>'; return; }
    tbody.innerHTML = incidents.map(i => `
      <tr>
        <td>${i.id}</td>
        <td>${esc(i.title)}</td>
        <td>${badge(i.severity)}</td>
        <td>${esc(i.status)}</td>
        <td>—</td>
        <td>${new Date(i.created_at).toLocaleString()}</td>
        <td><button class="btn btn-sm btn-primary" onclick="showIncidentDetail(${i.id})">Detail</button></td>
      </tr>
    `).join('');
  } catch (e) { /* optional */ }
}

async function showIncidentDetail(id) {
  try {
    const d = await api('/api/alerts/incidents/' + id + '/detail');
    const box = el('incident-detail');
    box.style.display = 'block';
    box.innerHTML = `
      <button class="close-btn" onclick="closeIncidentDetail()">✕</button>
      <h3>Incident #${d.incident.id} — ${esc(d.incident.title)}</h3>
      <p style="font-size:0.8rem;color:var(--text-dim)">${esc(d.incident.description || '')}</p>
      <div class="detail-grid">
        <div class="detail-item"><span class="label">Status</span><br>${esc(d.incident.status)}</div>
        <div class="detail-item"><span class="label">Highest severity</span><br>${badge(d.highest_severity)}</div>
        <div class="detail-item"><span class="label">Alerts</span><br>${d.total_alerts} (avg confidence ${d.avg_confidence != null ? (d.avg_confidence * 100).toFixed(1) + '%' : '—'})</div>
        <div class="detail-item"><span class="label">Attack types</span><br>${esc((d.attack_types || []).join(', '))}</div>
        <div class="detail-item"><span class="label">Sources</span><br>${esc((d.source_ips || []).join(', ') || '—')}</div>
        <div class="detail-item"><span class="label">Destinations</span><br>${esc((d.destination_ips || []).join(', ') || '—')}</div>
      </div>
      ${d.explanation ? `<h4 style="color:var(--accent);font-size:0.8rem;margin-top:12px">AI Summary</h4><p style="font-size:0.8rem;white-space:pre-wrap">${esc(d.explanation)}</p>` : ''}
      ${d.defensive_actions && d.defensive_actions.length ? `
        <h4 style="color:var(--accent);font-size:0.8rem;margin-top:12px">Recommended Defensive Actions</h4>
        <ul class="action-list">${d.defensive_actions.map(a => `
          <li class="priority-${esc(a.priority)}"><div class="action-name">${esc(a.action)}</div>
          <div class="action-reason">${esc(a.reason)}</div>
          <div class="action-meta">${esc(a.priority)} · ${a.approval_required ? 'approval required' : 'informational'}</div></li>
        `).join('')}</ul>` : ''}
      <h4 style="color:var(--accent);font-size:0.8rem;margin-top:12px">Related Alerts</h4>
      <table class="data-table"><thead><tr><th>ID</th><th>Attack</th><th>Severity</th><th>Source</th><th>Time</th></tr></thead><tbody>
      ${d.related_alerts.map(a => `<tr><td>${a.id}</td><td>${esc(a.attack_type)}</td><td>${badge(a.severity)}</td><td>${esc(a.src_ip)}</td><td>${new Date(a.timestamp).toLocaleString()}</td></tr>`).join('')}
      </tbody></table>
    `;
    box.scrollIntoView({ behavior: 'smooth' });
  } catch (e) { alert('Detail failed: ' + e.message); }
}
function closeIncidentDetail() { el('incident-detail').style.display = 'none'; }

async function loadAttackers() {
  try {
    const top = await api('/api/dashboard/top-attackers?limit=10&hours=24');
    const tbody = el('table-attackers').querySelector('tbody');
    if (!top.length) { tbody.innerHTML = '<tr><td colspan="4">No attackers.</td></tr>'; return; }
    tbody.innerHTML = top.map(a => `
      <tr>
        <td>${esc(a.src_ip)}</td>
        <td>${a.attack_count}</td>
        <td>${esc((a.attack_types || []).join(', '))}</td>
        <td>${badge(a.max_severity)}</td>
      </tr>`).join('');
  } catch (e) { /* optional */ }
}

function drawCharts(dist, timeline) {
  if (!charts.dist) {
    const ctxD = el('chart-distribution').getContext('2d');
    charts.dist = new Chart(ctxD, { type: 'bar', data: { labels: [], datasets: [{ label: 'Alerts', data: [], backgroundColor: '#58a6ff' }] }, options: { plugins: { legend: { display: false } } } });
    const ctxT = el('chart-timeline').getContext('2d');
    charts.timeline = new Chart(ctxT, { type: 'line', data: { labels: [], datasets: ['critical', 'high', 'medium', 'low'].map((k, i) => ({ label: k, data: [], borderColor: ['#ff4757', '#ff6b35', '#f0c929', '#58a6ff'][i], fill: false, tension: 0.2 })) }, options: { plugins: { legend: { labels: { color: '#8b949e' } } }, scales: { x: { ticks: { color: '#8b949e' } }, y: { ticks: { color: '#8b949e' } } } } });
  }
  charts.dist.data.labels = Object.keys(dist);
  charts.dist.data.datasets[0].data = Object.values(dist);
  charts.dist.update();

  charts.timeline.data.labels = timeline.map(t => t.timestamp.slice(5));
  ['critical', 'high', 'medium', 'low'].forEach((k, i) => {
    charts.timeline.data.datasets[i].data = timeline.map(t => t[k] || 0);
  });
  charts.timeline.update();
}

async function loadAnalytics() {
  try {
    const dist = await api('/api/dashboard/attack-distribution?hours=24');
    const timeline = await api('/api/dashboard/timeline?hours=24&interval_minutes=240');
    drawCharts(dist, timeline);
  } catch (e) { /* optional */ }
}

async function refreshAll() {
  loadStats(); loadRecentAlerts(); loadIncidents(); loadAttackers(); loadAnalytics();
}

/* Poll every 15s so dashboards stay live without a websocket. */
let booted = false;
function boot() {
  el('api-base').value = cfg.base;
  el('api-key').value = cfg.key;
  if (booted) { refreshAll(); return; }
  booted = true;
  refreshAll();
  setInterval(refreshAll, 15000);
  setInterval(async () => {
    try { await api('/health'); setStatus('ok', 'connected'); }
    catch (e) { setStatus('err', 'offline'); }
  }, 20000);
  setStatus('ok', 'connected');
}

document.addEventListener('DOMContentLoaded', () => {
  el('btn-save-config').addEventListener('click', saveConfig);
  el('btn-predict').addEventListener('click', runPrediction);
  el('btn-sample-attack').addEventListener('click', () => loadSample('sample_attack'));
  el('btn-sample-benign').addEventListener('click', () => loadSample('sample_benign'));
  boot();
});