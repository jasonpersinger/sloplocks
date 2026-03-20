# Frontend Redesign: Picks-Focused UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multi-section `index.html` with a focused picks feed showing 5 cards per view, cross-sport by default with per-sport tabs.

**Architecture:** Single-file `index.html` full rewrite. No pipeline or backend changes. All data comes from existing `data/{sport}/predictions.json` files. The JS is structured as plain functions inside an IIFE — same pattern as current code, no build step or framework.

**Tech Stack:** Vanilla HTML/CSS/JS, JetBrains Mono (Google Fonts), existing per-sport `predictions.json` data files.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `index.html` | **Full rewrite** | New HTML structure, all new CSS, all new JS |
| `sw.js` | **Bump cache version** | Increment `CACHE_NAME` string |
| `pipeline/*` | **No change** | Backend untouched |

---

## Task 1: HTML Skeleton + CSS

Build the static HTML shell and all CSS. No JavaScript logic yet — just the page structure and styles.

**Files:**
- Modify: `index.html` (full rewrite)

- [ ] **Step 1: Replace index.html with new skeleton**

Write the following as the complete new `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SLOP LOCKS</title>
  <meta name="description" content="Daily sports picks with model edge.">
  <meta name="theme-color" content="#39FF14">
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" type="image/png" sizes="192x192" href="/icons/icon-192.png">
  <link rel="apple-touch-icon" href="/icons/icon-192.png">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --slime: #39FF14;
      --dim: #888888;
      --dimmer: #555555;
      --hacker: #1A1A1A;
      --card-bg: #0d0d0d;
      --black: #000000;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      background: var(--black);
      color: #EAEAEA;
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.2;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .container {
      width: 100%;
      max-width: 450px;
      border-left: 1px solid var(--hacker);
      border-right: 1px solid var(--hacker);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      background: var(--black);
    }

    /* ── Header ── */
    .site-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 16px 12px 12px;
      border-bottom: 1px solid var(--hacker);
    }
    .site-logo {
      font-size: 14px;
      font-weight: 700;
      color: var(--slime);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .site-date {
      font-size: 10px;
      color: var(--dimmer);
      text-transform: uppercase;
    }

    /* ── Nav pills ── */
    .pill-nav {
      display: flex;
      border-bottom: 1px solid var(--hacker);
      overflow-x: auto;
    }
    .pill {
      background: none;
      border: none;
      border-right: 1px solid var(--hacker);
      color: var(--dimmer);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 10px 14px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
      outline: none;
      transition: color 0.15s;
    }
    .pill:last-child { border-right: none; }
    .pill.active { color: var(--slime); }
    .pill:hover:not(.active) { color: #888; }

    /* ── Pick feed ── */
    #pick-feed {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1px;
      background: var(--hacker);
      border-bottom: 1px solid var(--hacker);
    }

    /* ── Loading / empty / error states ── */
    .feed-message {
      background: var(--black);
      padding: 40px 16px;
      text-align: center;
      font-size: 11px;
      color: var(--dimmer);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .feed-message.pulse { animation: pulseAnim 2s infinite; }
    @keyframes pulseAnim { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* ── Pick card ── */
    .pick-card {
      background: var(--card-bg);
      border: 1px solid var(--hacker);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .card-sport-date {
      font-size: 9px;
      color: var(--dimmer);
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    .card-pick-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    .card-team {
      font-size: 18px;
      font-weight: 700;
      color: var(--slime);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .card-odds {
      font-size: 22px;
      font-weight: 700;
      color: #fff;
    }
    .card-opponent {
      font-size: 12px;
      color: var(--dim);
      text-transform: uppercase;
    }
    .card-stats {
      display: flex;
      gap: 18px;
      font-size: 10px;
    }
    .stat-model  { color: var(--slime); }
    .stat-market { color: var(--dim); }
    .stat-edge   { color: var(--slime); }

    /* ── Edge bar ── */
    .edge-labels {
      display: flex;
      justify-content: space-between;
      font-size: 9px;
      color: var(--dimmer);
      margin-bottom: 3px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .edge-bar-outer {
      width: 100%;
      height: 4px;
      background: var(--hacker);
      display: flex;
    }
    .edge-bar-market {
      height: 4px;
      background: #333;
      flex-shrink: 0;
    }
    .edge-bar-edge {
      height: 4px;
      background: var(--slime);
      flex-shrink: 0;
    }

    /* ── Blurb ── */
    .card-blurb {
      font-size: 10px;
      color: #666;
      line-height: 1.5;
      border-top: 1px dashed var(--hacker);
      padding-top: 8px;
    }

    /* ── Stats footer ── */
    .stats-footer {
      padding: 12px;
      text-align: center;
      font-size: 10px;
      color: var(--dimmer);
      border-top: 1px solid var(--hacker);
      margin-top: auto;
    }
  </style>
</head>
<body>
  <div class="container">

    <header class="site-header">
      <div class="site-logo">SLOP LOCKS</div>
      <div class="site-date" id="header-date"></div>
    </header>

    <nav class="pill-nav">
      <button class="pill active" data-sport="all">ALL</button>
      <button class="pill" data-sport="nba">NBA</button>
      <button class="pill" data-sport="ncaam">NCAAM</button>
      <button class="pill" data-sport="mlb">MLB</button>
      <button class="pill" data-sport="mma">MMA</button>
    </nav>

    <main id="pick-feed">
      <div class="feed-message pulse" id="loading-msg">LOADING...</div>
    </main>

    <footer class="stats-footer" id="stats-footer">-- · -- · ROI --</footer>

  </div>

  <script>
    // JS added in Tasks 2-5
    document.getElementById('header-date').textContent =
      new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase();
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify static layout in browser**

Serve from repo root: `python3 -m http.server 8080` then open `http://localhost:8080`. Confirm:
- "SLOP LOCKS" top-left in slime green, today's date top-right in dim
- Pill nav: ALL (slime green), NBA NCAAM MLB MMA (all dim)
- "LOADING..." message pulsing in feed area
- "-- · -- · ROI --" footer at bottom

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: new index.html skeleton with header, pill nav, and CSS"
```

---

## Task 2: Formatting Utilities + Pick Card Renderer

Add the JS helper functions and `createPickCard(pick)`. The page still shows the static loading state — this task just provides the building blocks.

**Note:** All DOM manipulation uses `createElement` / `appendChild` / `textContent` — never `innerHTML` with dynamic data.

**Files:**
- Modify: `index.html` — replace the `<script>` block

- [ ] **Step 1: Replace the script block with the IIFE + utilities + card renderer**

Replace everything between `<script>` and `</script>` with:

```javascript
(function () {
  'use strict';

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(function () {});
  }

  // ── Date header ──────────────────────────────────────────────────────────
  document.getElementById('header-date').textContent =
    new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase();

  // ── Formatting helpers ───────────────────────────────────────────────────
  function formatOdds(american) {
    if (american == null) return '--';
    return american >= 0 ? '+' + american : String(american);
  }

  function formatDate(iso) {
    // "2026-03-20" → "MAR 20"
    // Use noon UTC to avoid timezone off-by-one for dates near midnight
    var d = new Date(iso + 'T12:00:00Z');
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' }).toUpperCase();
  }

  var SPORT_LABELS = { nba: 'NBA', ncaam: 'NCAAM', mlb: 'MLB', mma: 'MMA' };
  function formatSportLabel(sport) {
    return SPORT_LABELS[sport] || sport.toUpperCase();
  }

  function pct(p) { return (p * 100).toFixed(1) + '%'; }

  // ── createPickCard ────────────────────────────────────────────────────────
  // pick: a slop_locks item with .sport tagged on it
  function createPickCard(pick) {
    var pickedTeam = pick.pick === 'home' ? pick.home_team : pick.away_team;
    var opponent   = pick.pick === 'home' ? pick.away_team : pick.home_team;

    var card = document.createElement('div');
    card.className = 'pick-card';

    // Row 1: sport · date
    var sportDate = document.createElement('div');
    sportDate.className = 'card-sport-date';
    sportDate.textContent = formatSportLabel(pick.sport) + ' · ' + formatDate(pick.date);
    card.appendChild(sportDate);

    // Row 2: picked team name (left) + odds (right)
    var pickRow = document.createElement('div');
    pickRow.className = 'card-pick-row';

    var teamEl = document.createElement('div');
    teamEl.className = 'card-team';
    teamEl.textContent = pickedTeam.toUpperCase();
    pickRow.appendChild(teamEl);

    var oddsEl = document.createElement('div');
    oddsEl.className = 'card-odds';
    oddsEl.textContent = formatOdds(pick.american_odds);
    pickRow.appendChild(oddsEl);

    card.appendChild(pickRow);

    // Row 3: vs OPPONENT
    var oppEl = document.createElement('div');
    oppEl.className = 'card-opponent';
    oppEl.textContent = 'vs ' + opponent.toUpperCase();
    card.appendChild(oppEl);

    // Row 4: Model % · Market % · Edge +%
    var statsRow = document.createElement('div');
    statsRow.className = 'card-stats';

    var modelEl = document.createElement('span');
    modelEl.className = 'stat-model';
    modelEl.textContent = 'Model: ' + pct(pick.model_prob);
    statsRow.appendChild(modelEl);

    var marketEl = document.createElement('span');
    marketEl.className = 'stat-market';
    marketEl.textContent = 'Market: ' + pct(pick.implied_prob);
    statsRow.appendChild(marketEl);

    var edgeEl = document.createElement('span');
    edgeEl.className = 'stat-edge';
    edgeEl.textContent = 'Edge: +' + pct(pick.edge);
    statsRow.appendChild(edgeEl);

    card.appendChild(statsRow);

    // Row 5: edge bar labels + bar
    var labelsEl = document.createElement('div');
    labelsEl.className = 'edge-labels';

    var leftLabel = document.createElement('span');
    leftLabel.textContent = 'MARKET IMPLIED';
    labelsEl.appendChild(leftLabel);

    var rightLabel = document.createElement('span');
    rightLabel.textContent = 'MODEL EDGE';
    labelsEl.appendChild(rightLabel);

    card.appendChild(labelsEl);

    var barOuter = document.createElement('div');
    barOuter.className = 'edge-bar-outer';

    var barMarket = document.createElement('div');
    barMarket.className = 'edge-bar-market';
    barMarket.style.width = (pick.implied_prob * 100).toFixed(2) + '%';
    barOuter.appendChild(barMarket);

    var barEdge = document.createElement('div');
    barEdge.className = 'edge-bar-edge';
    barEdge.style.width = (pick.edge * 100).toFixed(2) + '%';
    barOuter.appendChild(barEdge);

    card.appendChild(barOuter);

    // Row 6: AI blurb (only if non-empty)
    if (pick.blurb && pick.blurb.trim() !== '') {
      var blurbEl = document.createElement('div');
      blurbEl.className = 'card-blurb';
      blurbEl.textContent = pick.blurb;
      card.appendChild(blurbEl);
    }

    return card;
  }

  // Tasks 3-5 JS goes here

  loadAllSports(); // init — defined in Task 3

})();
```

- [ ] **Step 2: Verify no console errors**

Reload the page. Confirm:
- No JS errors in console (the `loadAllSports` call will error — that's expected until Task 3)
- Service worker registers without error

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add pick card renderer and formatting utilities"
```

---

## Task 3: ALL View — loadAllSports() + renderFeed() + renderStats()

Implement the cross-sport fetch, feed rendering, and stats footer. This makes the ALL tab fully functional.

**Files:**
- Modify: `index.html` — replace `// Tasks 3-5 JS goes here`

- [ ] **Step 1: Add feed state, renderFeed, renderStats, and loadAllSports**

Replace `// Tasks 3-5 JS goes here` with:

```javascript
  // ── Feed state ───────────────────────────────────────────────────────────
  var feed = document.getElementById('pick-feed');
  var statsFooter = document.getElementById('stats-footer');
  var currentSport = 'all';
  var loadedSportsData = {}; // { nba: predictionsJSON, ncaam: ..., ... }

  // ── clearFeed ─────────────────────────────────────────────────────────────
  function clearFeed() {
    while (feed.firstChild) {
      feed.removeChild(feed.firstChild);
    }
  }

  // ── showMessage ───────────────────────────────────────────────────────────
  function showMessage(text, pulse) {
    clearFeed();
    var el = document.createElement('div');
    el.className = pulse ? 'feed-message pulse' : 'feed-message';
    el.textContent = text;
    feed.appendChild(el);
  }

  // ── renderFeed ────────────────────────────────────────────────────────────
  // picks: array of tagged slop_locks items (each has .sport set)
  function renderFeed(picks) {
    clearFeed();
    if (!picks || picks.length === 0) {
      showMessage('NO PICKS TODAY — CHECK BACK AFTER 6AM UTC', false);
      return;
    }
    picks.forEach(function (pick) {
      feed.appendChild(createPickCard(pick));
    });
  }

  // ── renderStats ───────────────────────────────────────────────────────────
  function renderStats(mode) {
    if (mode === 'all') {
      var wins = 0, losses = 0, evaluated = 0;
      Object.keys(loadedSportsData).forEach(function (sport) {
        var ps = (loadedSportsData[sport].pick_stats || {}).all || {};
        wins      += ps.wins      || 0;
        losses    += ps.losses    || 0;
        evaluated += ps.evaluated || 0;
      });
      var hitRate = evaluated > 0 ? wins / evaluated : null;
      if (hitRate === null) {
        statsFooter.textContent = '-- · -- · ROI --';
      } else {
        statsFooter.textContent =
          wins + '-' + losses + ' · ' + (hitRate * 100).toFixed(1) + '% · ROI --';
      }
      return;
    }

    // Per-sport
    var data = loadedSportsData[mode];
    if (!data) { statsFooter.textContent = '-- · -- · ROI --'; return; }
    var ps = (data.pick_stats || {}).all || {};
    var hitRate = ps.hit_rate != null ? ps.hit_rate : null;
    var roi     = ps.roi      != null ? ps.roi      : null;
    if (hitRate === null) {
      statsFooter.textContent = '-- · -- · ROI --';
    } else {
      var roiStr = roi != null
        ? 'ROI ' + (roi >= 0 ? '+' : '') + (roi * 100).toFixed(1) + '%'
        : 'ROI --';
      statsFooter.textContent =
        (ps.wins || 0) + '-' + (ps.losses || 0) + ' · ' +
        (hitRate * 100).toFixed(1) + '% · ' + roiStr;
    }
  }

  // ── loadAllSports ─────────────────────────────────────────────────────────
  var ALL_SPORTS = ['nba', 'ncaam', 'mlb', 'mma'];

  function loadAllSports() {
    currentSport = 'all';
    loadedSportsData = {};
    showMessage('LOADING...', true);

    var fetches = ALL_SPORTS.map(function (sport) {
      return fetch('data/' + sport + '/predictions.json?t=' + Date.now())
        .then(function (res) { return res.json(); })
        .then(function (data) { return { sport: sport, data: data }; });
    });

    Promise.allSettled(fetches).then(function (results) {
      var allPicks = [];

      results.forEach(function (result) {
        if (result.status !== 'fulfilled') return; // skip failed fetches silently
        var sport = result.value.sport;
        var data  = result.value.data;
        loadedSportsData[sport] = data;

        (data.slop_locks || []).forEach(function (pick) {
          pick.sport = sport; // tag with source sport key
          allPicks.push(pick);
        });
      });

      if (Object.keys(loadedSportsData).length === 0) {
        showMessage('ERROR LOADING DATA', false);
        statsFooter.textContent = '-- · -- · ROI --';
        return;
      }

      // Sort by model_prob descending, take top 5
      allPicks.sort(function (a, b) { return b.model_prob - a.model_prob; });
      renderFeed(allPicks.slice(0, 5));
      renderStats('all');
    });
  }

  // Tasks 4-5 JS goes here
```

- [ ] **Step 2: Verify ALL view works end-to-end**

Reload `http://localhost:8080`. Confirm:
- "LOADING..." shows briefly, then cards render (or NO PICKS message if `slop_locks` is empty in all sport files)
- Each card shows: sport badge, team vs opponent, odds, model/market/edge stats, edge bar
- Footer shows aggregated record (or `-- · -- · ROI --` if no evaluated picks)
- No JS errors in console

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: implement ALL view with parallel fetch and top-5 cross-sport picks"
```

---

## Task 4: Per-Sport View — loadSportData() + Nav Wiring

Implement single-sport data loading and wire up all pill click handlers.

**Files:**
- Modify: `index.html` — replace `// Tasks 4-5 JS goes here`

- [ ] **Step 1: Add loadSportData() and nav pill listeners**

Replace `// Tasks 4-5 JS goes here` with:

```javascript
  // ── loadSportData ─────────────────────────────────────────────────────────
  function loadSportData(sport) {
    currentSport = sport;
    showMessage('LOADING...', true);

    fetch('data/' + sport + '/predictions.json?t=' + Date.now())
      .then(function (res) { return res.json(); })
      .then(function (data) {
        loadedSportsData[sport] = data;
        var locks = data.slop_locks || [];
        locks.forEach(function (pick) { pick.sport = sport; });
        renderFeed(locks);
        renderStats(sport);
      })
      .catch(function () {
        showMessage('ERROR LOADING DATA', false);
        statsFooter.textContent = '-- · -- · ROI --';
      });
  }

  // ── Nav pill event listeners ──────────────────────────────────────────────
  document.querySelectorAll('.pill').forEach(function (pill) {
    pill.addEventListener('click', function () {
      var sport = this.getAttribute('data-sport');
      if (sport === currentSport) return; // no-op if already active

      document.querySelectorAll('.pill').forEach(function (p) {
        p.classList.remove('active');
      });
      this.classList.add('active');

      if (sport === 'all') {
        loadAllSports();
      } else {
        loadSportData(sport);
      }
    });
  });
```

- [ ] **Step 2: Full nav verification**

Test all interactions:
1. Page load → ALL active (slime), others dim, cross-sport picks shown
2. Click NBA → NBA pill slime green, others dim, NBA picks shown, footer updates
3. Click ALL → returns to cross-sport view
4. Click NCAAM, MLB, MMA each in turn → correct picks shown per sport
5. Click the already-active pill → nothing happens (no reload/flicker)
6. Footer stats update on each tab switch
7. Zero JS errors in console

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: wire per-sport views and nav pill switching"
```

---

## Task 5: Cleanup + Service Worker Cache Bump

Audit for old artifacts, bump the SW cache version, and do a final end-to-end check.

**Files:**
- Modify: `index.html` — audit only
- Modify: `sw.js` — bump `CACHE_NAME`

- [ ] **Step 1: Audit index.html for old code artifacts**

Search the file for any of these strings — none should exist in the new file:
- `ticker-top`
- `slimegrinder`
- `longslop`
- `reveal-btn`
- `slate-section`
- `info-toggle`
- `match-row`
- `loadSportData('nba')` (old default — now replaced by `loadAllSports()`)

If any survive, remove them.

- [ ] **Step 2: Bump sw.js CACHE_NAME**

Open `sw.js`. Find the `CACHE_NAME` constant (e.g. `const CACHE_NAME = 'sloplocks-v4'`). Increment the version number by 1. This forces browsers to pick up the new `index.html` instead of serving a cached copy.

- [ ] **Step 3: Final end-to-end verification**

With `python3 -m http.server 8080` from repo root:
1. Open `http://localhost:8080` in a fresh incognito tab
2. Confirm: logo + today's date header, ALL active, picks or NO PICKS TODAY message
3. Confirm: no old sections (ticker, slimegrinder section label, full slate button, system logs)
4. Confirm: sport tab switching works correctly
5. Open DevTools → Application → Service Workers → confirm SW registered

- [ ] **Step 4: Final commit**

```bash
git add index.html sw.js
git commit -m "feat: frontend redesign complete — picks-focused UI with cross-sport ALL view"
```
