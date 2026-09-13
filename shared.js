/* ═══════════════════════════════════════════
   Satyam's Store — shared.js
   Backend: Supabase REST API (direct, no SDK)
   ═══════════════════════════════════════════ */

const SB_URL = 'https://jrmctduwylpeicjcbmqs.supabase.co';
const SB_KEY = 'sb_publishable_E3W5FNr_zAmej5fLElsvCA_OeDkde6L';

/* ── Supabase REST helpers ─────────────────────────────── */

function _sbHeaders(extras = {}) {
  // Use authenticated JWT if available (admin), fall back to anon key (store)
  let token = SB_KEY;
  try {
    const raw = sessionStorage.getItem('sb_session');
    if (raw) {
      const s = JSON.parse(raw);
      if (s.access_token) token = s.access_token;
    }
  } catch {}
  return {
    'apikey':        SB_KEY,
    'Authorization': 'Bearer ' + token,
    'Content-Type':  'application/json',
    'Prefer':        'return=representation',
    ...extras,
  };
}

// GET  /rest/v1/<table>?<query>
async function sbGet(tableAndQuery) {
  let token = SB_KEY;
  try {
    const raw = sessionStorage.getItem('sb_session');
    if (raw) { const s = JSON.parse(raw); if (s.access_token) token = s.access_token; }
  } catch {}
  const res = await fetch(
    `${SB_URL}/rest/v1/${tableAndQuery}`,
    {
      headers: {
        'apikey':        SB_KEY,
        'Authorization': 'Bearer ' + token,
      }
    }
  );
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    if (res.status === 401 || res.status === 403) {
      throw new Error(
        'Supabase auth failed (' + res.status + '). ' +
        'The API key may be wrong. Go to Supabase dashboard → Settings → API ' +
        'and copy the anon/public key (starts with eyJ). ' +
        'Update SB_KEY in shared.js.'
      );
    }
    throw new Error('sbGet ' + res.status + ': ' + txt.slice(0, 200));
  }
  return res.json();
}

// POST /rest/v1/<table>  — insert one row, returns inserted row
async function sbPost(table, body) {
  const res = await fetch(
    `${SB_URL}/rest/v1/${table}`,
    { method: 'POST', headers: _sbHeaders(), body: JSON.stringify(body) }
  );
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`sbPost ${res.status} ${txt}`);
  }
  return res.json();
}

// PATCH /rest/v1/<table>?<filter>  — update matching rows
async function sbPatch(tableAndFilter, body) {
  const res = await fetch(
    `${SB_URL}/rest/v1/${tableAndFilter}`,
    { method: 'PATCH', headers: _sbHeaders({ 'Prefer': 'return=minimal' }), body: JSON.stringify(body) }
  );
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`sbPatch ${res.status} ${txt}`);
  }
  // return=minimal gives 204 No Content — no body to parse
  return true;
}

// DELETE /rest/v1/<table>?<filter>
async function sbDelete(tableAndFilter) {
  const res = await fetch(
    `${SB_URL}/rest/v1/${tableAndFilter}`,
    { method: 'DELETE', headers: _sbHeaders({ 'Prefer': 'return=minimal' }) }
  );
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`sbDelete ${res.status} ${txt}`);
  }
  return true;
}

/* ── Store config (settings table) ────────────────────── */

const _DEFAULT_CONFIG = { storeStatus: 'open', autoCloseEnabled: false, autoCloseTime: '03:00', lastAutoCloseDate: '' };

async function getStoreConfig() {
  try {
    const data = await sbGet('settings?key=eq.store_config&select=value');
    if (data && data.length > 0) {
      return { ..._DEFAULT_CONFIG, ...(data[0].value || {}) };
    }
    // Row doesn't exist yet — create it silently
    try {
      await sbPost('settings', { key: 'store_config', value: _DEFAULT_CONFIG });
    } catch (_) {}
    return { ..._DEFAULT_CONFIG };
  } catch (error) {
    /* suppressed */
    return { ..._DEFAULT_CONFIG };
  }
}

async function setStoreConfig(patch) {
  const currentConfig = await getStoreConfig();
  const newConfig = { ...currentConfig, ...patch };
  try {
    await sbPatch('settings?key=eq.store_config', { value: newConfig });
  } catch (e) {
    try {
      await sbPost('settings', { key: 'store_config', value: newConfig });
    } catch (_) {}
  }
  return newConfig;
}

/* ── Cache layer ───────────────────────────────────────── */

const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function _cacheKey(label) {
  return 'ss_cache__' + label;
}
function _cacheGet(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const { ts, data } = JSON.parse(raw);
    if (Date.now() - ts > CACHE_TTL) { localStorage.removeItem(key); return null; }
    return data;
  } catch { return null; }
}
function _cacheSet(key, data) {
  try { localStorage.setItem(key, JSON.stringify({ ts: Date.now(), data })); } catch {}
}
function _cacheInvalidate(label) {
  try {
    const prefix = 'ss_cache__' + label;
    Object.keys(localStorage).forEach(k => { if (k.startsWith(prefix)) localStorage.removeItem(k); });
  } catch {}
}

/* ── Product helpers ───────────────────────────────────── */

// Supabase products rows -> internal product objects
function parseProducts(rows) {
  if (!Array.isArray(rows)) return [];
  return rows
    .map(r => ({
      id:             String(r.id   ?? '').trim(),
      name:           String(r.name ?? '').trim(),
      price:          parseFloat(r.price)   || 0,
      stock:          parseInt(r.stock, 10) || 0,
      image:          String(r.image ?? '').trim(),
      purchase_price: parseFloat(r.purchase_price) || 0,
      hidden: r.hidden === true || r.hidden === 'true',
      category: String(r.category || 'others').trim(),
    }))
    .filter(p => p.id && p.name);
}

/* ── Utility ───────────────────────────────────────────── */

function esc(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

let _tt;
function toast(msg, dur = 3000) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(_tt);
  _tt = setTimeout(() => el.classList.add('hidden'), dur);
}

let _loaderHidden = false;
function hideLoader() {
  if (_loaderHidden) return;
  _loaderHidden = true;
  const el = document.getElementById('page-loader');
  if (!el) return;
  // Stop all animations immediately so they don't keep running under the fade
  el.style.animation = 'none';
  const card = el.querySelector('.animating-card');
  if (card) card.style.animation = 'none';
  el.classList.add('fade-out');
  setTimeout(() => { el.style.display = 'none'; }, 540);
}

// Format Supabase timestamp -> IST time string (HH:MM AM/PM)
// Supabase timestamps may come with or without timezone info.
// Strategy: normalise to UTC epoch, then add IST offset (+5:30), read UTC fields.
function cleanSheetVal(val) {
  if (!val) return '';
  try {
    const ms = _toUTCms(val);
    if (ms === null) return String(val);
    // Shift to IST: UTC + 5h30m = +19800000ms
    const ist = new Date(ms + 19800000);
    const hh  = ist.getUTCHours();
    const min = String(ist.getUTCMinutes()).padStart(2, '0');
    const ampm = hh >= 12 ? 'PM' : 'AM';
    const h12  = (hh % 12) || 12;
    return String(h12).padStart(2, '0') + ':' + min + ' ' + ampm;
  } catch { return String(val); }
}

// Return IST date string "DD/MM/YYYY" from a Supabase timestamp
function istDateStr(val) {
  if (!val) return '';
  try {
    const ms = _toUTCms(val);
    if (ms === null) return String(val);
    const ist = new Date(ms + 19800000);
    return String(ist.getUTCDate()).padStart(2, '0') + '/' +
           String(ist.getUTCMonth() + 1).padStart(2, '0') + '/' +
           ist.getUTCFullYear();
  } catch { return String(val); }
}

// Parse any Supabase timestamp to a UTC epoch ms value.
// Handles: "2026-04-21T10:24:00+00:00", "2026-04-21T10:24:00Z",
//          "2026-04-21T10:24:00.123456+05:30", "2026-04-21T10:24:00" (bare, assumed UTC)
function _toUTCms(val) {
  const s = String(val).trim();
  // Already has timezone info — let the engine parse it (unambiguous)
  if (/[Zz]$/.test(s) || /[+-]\d{2}:\d{2}$/.test(s)) {
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d.getTime();
  }
  // Bare timestamp with no timezone — treat as UTC by appending Z
  const d = new Date(s.replace(' ', 'T') + 'Z');
  return isNaN(d.getTime()) ? null : d.getTime();
}

function placeholderSVG() {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="1.2" opacity=".25"><rect x="2" y="7" width="20" height="14" rx="2"/><circle cx="12" cy="14" r="3"/><path d="M2 10h20"/></svg>';
}
