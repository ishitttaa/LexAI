// ── TAB SWITCHING ──
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    
    if (btn.dataset.tab === 'tab-signals') loadSignalsAndTrail();
  });
});


// ── TAB 1: QA EVALUATOR ──
const btnQaDemo = document.getElementById('btn-qa-demo');
const btnQaAudit = document.getElementById('btn-qa-audit');

function toggleCustomContext() {
  const wrap = document.getElementById('custom-context-wrap');
  const btn  = document.getElementById('btn-ctx-toggle');
  const hidden = wrap.classList.toggle('hidden');
  btn.textContent = hidden
    ? '⚙️ Advanced: Use Custom Context instead of auto-fetch'
    : '✕ Hide Custom Context (use auto-fetch)';
}

btnQaDemo.addEventListener('click', () => {
  document.getElementById('qa-question').value = "What is the capital of Australia, and who is the Prime Minister?";
  document.getElementById('qa-answer').value = "The capital of Australia is Sydney, and the Prime Minister is Anthony Albanese.";
  document.getElementById('qa-topic').value = "Geography";
  // Show context panel and fill it for demo
  const wrap = document.getElementById('custom-context-wrap');
  if (wrap.classList.contains('hidden')) toggleCustomContext();
  document.getElementById('qa-context').value = "Australia's capital city is Canberra, not Sydney. Anthony Albanese is the current Prime Minister of Australia.";
});

btnQaAudit.addEventListener('click', async () => {
  const q = document.getElementById('qa-question').value.trim();
  const a = document.getElementById('qa-answer').value.trim();
  const t = document.getElementById('qa-topic').value.trim();
  // Context is optional — only send if user filled it in
  const ctxEl = document.getElementById('qa-context');
  const c = ctxEl ? ctxEl.value.trim() : '';

  if (!q || !a) return alert('Please fill in Question and LLM Answer.');

  btnQaAudit.textContent = c
    ? "Running 4-Method Audit..." 
    : "🌐 Fetching context from web + Running Audit...";
  btnQaAudit.disabled = true;

  try {
    const res = await fetch('/api/qa-audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, context: c || null, answer: a, topic: t })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }
    const data = await res.json();
    renderQaResults(data);
  } catch (err) {
    alert("Audit failed: " + err.message);
  } finally {
    btnQaAudit.textContent = "🔍 Run QA Evaluation";
    btnQaAudit.disabled = false;
  }
});

function renderQaResults(data) {
  document.getElementById('qa-results').classList.remove('hidden');
  
  // Summary
  document.getElementById('qa-val-score').textContent = data.ensemble_score.toFixed(2);
  document.getElementById('qa-val-score').style.color = getScoreColor(data.ensemble_score);
  
  const vEl = document.getElementById('qa-val-verdict');
  vEl.textContent = data.verdict.toUpperCase();
  vEl.style.color = data.verdict === 'hallucination' ? 'var(--hallucination)' : (data.verdict === 'plausible' ? 'var(--plausible)' : 'var(--verified)');
  
  document.getElementById('qa-val-time').textContent = data.processing_ms + ' ms';

  // Context Sources Panel — show if auto-fetched
  const sourcesPanel = document.getElementById('qa-context-sources');
  if (data.context_sources && data.context_sources.length > 0) {
    sourcesPanel.classList.remove('hidden');
    document.getElementById('qa-source-links').innerHTML = data.context_sources.map(s =>
      `<a href="${s.url}" target="_blank" class="ctx-source-link">
        <span class="ctx-source-icon">🔗</span>
        <span>${escapeHtml(s.title)}</span>
      </a>`
    ).join('');
    if (data.fetched_context) {
      document.getElementById('qa-fetched-context-text').textContent = data.fetched_context;
    }
  } else {
    sourcesPanel.classList.add('hidden');
  }

  // Method Cards
  const methods = [data.embedding_score, data.groundedness_score, data.consistency_score, data.geval_score];
  const container = document.getElementById('method-cards-container');
  container.innerHTML = methods.map(m => `
    <div class="method-card">
      <div class="method-header">
        <span>${m.name}</span>
        <span class="method-score" style="color: ${getScoreColor(m.score)}">${m.score.toFixed(2)}</span>
      </div>
      <div class="method-explanation">${m.explanation}</div>
      <div style="font-size:0.75rem; color:var(--text-muted); margin-top:8px;">Weight in ensemble: ${(m.weight*100).toFixed(0)}%</div>
    </div>
  `).join('');
}

function getScoreColor(score) {
  if (score < 0.35) return 'var(--verified)';
  if (score < 0.60) return 'var(--plausible)';
  return 'var(--hallucination)';
}


// ── TAB 2: DOCUMENT AUDITOR ──
const btnDocAudit = document.getElementById('btn-doc-audit');
const docInput = document.getElementById('doc-input');
const docResults = document.getElementById('doc-results');
const annotatedDoc = document.getElementById('annotated-doc');
const claimsList = document.getElementById('claims-list');

const pdfUpload = document.getElementById('pdf-upload');
const pdfUploadStatus = document.getElementById('pdf-upload-status');

if (pdfUpload) {
  pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    pdfUploadStatus.textContent = "Extracting text from PDF...";
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch('/api/extract-pdf', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to extract PDF');
      }
      const data = await res.json();
      docInput.value = data.text;
      pdfUploadStatus.textContent = "Text extracted successfully!";
      setTimeout(() => { pdfUploadStatus.textContent = ""; }, 3000);
    } catch (err) {
      pdfUploadStatus.textContent = "Error: " + err.message;
    }
    pdfUpload.value = ''; // Reset input
  });
}

if (btnDocAudit) {
  btnDocAudit.addEventListener('click', async () => {
    const docText = docInput.value.trim();
    if (!docText) return alert('Please paste a document first.');

    btnDocAudit.textContent = "Auditing Document (Groq + Search)...";
    btnDocAudit.disabled = true;
    docResults.classList.add('hidden');

    try {
      const res = await fetch('/api/doc-audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document: docText })
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Server error");
      }

      const data = await res.json();
      renderDocResults(data);
    } catch (err) {
      alert("Document audit failed: " + err.message);
    } finally {
      btnDocAudit.textContent = "Run Document Audit";
      btnDocAudit.disabled = false;
    }
  });
}

function renderDocResults(data) {
  docResults.classList.remove('hidden');
  
  let docHTML = escapeHtml(data.document);
  
  // Sort claims by original_text length descending to avoid subset replacement issues
  const sortedClaims = [...data.claims].sort((a, b) => b.original_text.length - a.original_text.length);
  
  const placeholders = [];
  sortedClaims.forEach((claim) => {
    const origEscaped = escapeHtml(claim.original_text);
    if (!origEscaped) return;
    
    const escapedRegex = origEscaped.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
    const regex = new RegExp(escapedRegex, 'g');
    
    docHTML = docHTML.replace(regex, (match) => {
      const placeholder = `___CLAIM_PLACEHOLDER_${placeholders.length}___`;
      placeholders.push({
        placeholder,
        html: `<span class="doc-highlight highlight-${claim.status}" data-claim-id="${claim.id}">${match}</span>`
      });
      return placeholder;
    });
  });
  
  placeholders.forEach(p => {
    docHTML = docHTML.replace(p.placeholder, p.html);
  });
  
  annotatedDoc.innerHTML = docHTML;
  
  claimsList.innerHTML = data.claims.map(c => {
    const statusClass = `bg-${c.status}`;
    const sourcesHTML = c.sources && c.sources.length > 0
      ? `<div class="claim-sources">
           <div class="claim-sources-title">Sources:</div>
           ${c.sources.map(s => `<a href="${s.url}" target="_blank" class="source-link">🔗 ${escapeHtml(s.title || s.url)}</a>`).join('')}
         </div>`
      : `<div class="claim-sources"><span class="text-muted">No sources found</span></div>`;
      
    return `
      <div class="claim-item" id="claim-item-${c.id}" data-claim-id="${c.id}">
        <span class="claim-status ${statusClass}">${c.status} (Conf: ${(c.confidence * 100).toFixed(0)}%)</span>
        <div class="claim-text">"${escapeHtml(c.claim_text)}"</div>
        <div class="claim-explanation">${escapeHtml(c.explanation)}</div>
        ${sourcesHTML}
      </div>
    `;
  }).join('');
  
  // Add interactivity
  document.querySelectorAll('.doc-highlight').forEach(el => {
    el.addEventListener('click', () => {
      const claimId = el.getAttribute('data-claim-id');
      highlightClaimInSidebar(claimId);
    });
  });
  
  document.querySelectorAll('.claim-item').forEach(el => {
    el.addEventListener('click', () => {
      const claimId = el.getAttribute('data-claim-id');
      highlightClaimInDoc(claimId);
    });
  });
}

function highlightClaimInSidebar(claimId) {
  document.querySelectorAll('.claim-item').forEach(item => item.classList.remove('active'));
  const target = document.getElementById(`claim-item-${claimId}`);
  if (target) {
    target.classList.add('active');
    target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function highlightClaimInDoc(claimId) {
  document.querySelectorAll('.doc-highlight').forEach(h => h.classList.remove('active-highlight'));
  const highlights = document.querySelectorAll(`.doc-highlight[data-claim-id="${claimId}"]`);
  highlights.forEach(h => {
    h.classList.add('active-highlight');
  });
  if (highlights.length > 0) {
    highlights[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


// ── TAB 3: SIGNALS & AUDIT TRAIL ──
document.getElementById('btn-refresh-trail').addEventListener('click', loadSignalsAndTrail);

async function loadSignalsAndTrail() {
  // Load Signals
  const sigRes = await fetch('/api/signals');
  const signals = await sigRes.json();
  
  const sigContainer = document.getElementById('signals-container');
  if (signals.patterns.length === 0) {
    sigContainer.innerHTML = '<div class="text-muted">No clear failure patterns detected yet.</div>';
  } else {
    sigContainer.innerHTML = signals.patterns.map(p => `
      <div class="signal-item">
        <div class="signal-label">${p.label}</div>
        <div class="signal-desc">${p.description}</div>
      </div>
    `).join('');
  }

  // Plotly Distribution Chart
  const dist = signals.score_distribution;
  Plotly.newPlot('chart-distribution', [{
    x: ['Low Risk (Verified)', 'Medium Risk (Plausible)', 'High Risk (Hallucination)'],
    y: [dist.low, dist.medium, dist.high],
    type: 'bar',
    marker: { color: ['#22c55e', '#f59e0b', '#ef4444'] }
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#9ca3af' },
    margin: { t: 10, b: 30, l: 30, r: 10 }
  }, {displayModeBar: false});

  // Load Audit Trail Table
  const trRes = await fetch('/api/audit-trail');
  const trail = await trRes.json();
  
  const tbody = document.getElementById('audit-tbody');
  tbody.innerHTML = trail.records.map(r => `
    <tr>
      <td>#${r.id}</td>
      <td>${new Date(r.timestamp).toLocaleString()}</td>
      <td>${r.topic}</td>
      <td title="${r.question}">${r.question.substring(0,40)}...</td>
      <td style="color:${getScoreColor(r.ensemble_score)}; font-weight:bold;">${r.ensemble_score.toFixed(2)}</td>
      <td><span class="verdict-badge bg-${r.verdict}">${r.verdict}</span></td>
    </tr>
  `).join('');
}


// Load trail initially if needed, though tab click handles it.
