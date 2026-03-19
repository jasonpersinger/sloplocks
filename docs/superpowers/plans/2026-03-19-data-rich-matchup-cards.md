# Data-Rich Matchup Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance matchup cards to show win probability and edge for both teams inline with their names.

**Architecture:** 
- CSS update to `.team-row` to support a 3-zone layout (Name, Stats, Badge).
- JS update to `createMatchCard` to inject the new stats cluster.
- Mobile-first approach with ellipsis for long team names.

**Tech Stack:** HTML, CSS, Vanilla JS.

---

### Task 1: Update CSS for Data Density

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Update `.team-row` and add `.team-stats` styles.**
  Add the following to the `<style>` block in `index.html`:
```css
    .team-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      min-height: 1.5rem;
    }
    .team-name {
      font-family: var(--font-heading);
      font-size: 1.1rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .team-stats {
      display: flex;
      gap: 0.5rem;
      font-family: var(--font-mono);
      font-size: 0.7rem;
      white-space: nowrap;
      color: var(--text-dim);
      min-width: 90px;
      justify-content: flex-end;
    }
    .team-stats .prob { color: var(--text); }
    .team-stats .edge.pos { color: var(--slime); }
    .team-stats .edge.neg { color: var(--text-muted); }
    .pick-badge {
      min-width: 45px;
      text-align: center;
    }
```

- [ ] **Step 2: Commit.**
```bash
git add index.html
git commit -m "feat(ui): add CSS for data-rich team rows"
```

### Task 2: Refactor JavaScript for Dual-Sided Data

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Rewrite the team row rendering in `createMatchCard`.**
  Locate the section building `hRow` and `aRow` and update it to calculate win % and edge for both sides.
  
```javascript
        function createStats(side) {
          var p = probs[side];
          var e = (edges[side] && edges[side].edge) ? edges[side].edge : 0;
          var div = createEl('div', 'team-stats');
          div.appendChild(createEl('span', 'prob', (p * 100).toFixed(1) + '%'));
          var edgeSpan = createEl('span', 'edge ' + (e > 0 ? 'pos' : 'neg'), '(' + (e > 0 ? '+' : '') + (e * 100).toFixed(1) + '%)');
          div.appendChild(edgeSpan);
          return div;
        }

        var hRow = createEl('div', 'team-row');
        hRow.appendChild(createEl('span', 'team-name', match.home_team));
        hRow.appendChild(createStats('home'));
        hRow.appendChild(createEl('span', 'pick-badge' + (match.pick === 'home' ? ' active' : ''), 'WIN'));
        
        var aRow = createEl('div', 'team-row');
        aRow.appendChild(createEl('span', 'team-name', match.away_team));
        aRow.appendChild(createStats('away'));
        aRow.appendChild(createEl('span', 'pick-badge' + (match.pick === 'away' ? ' active' : ''), 'WIN'));
```

- [ ] **Step 2: Commit.**
```bash
git add index.html
git commit -m "feat(ui): render win probability and edge for both teams"
```

### Task 3: Final Verification

- [ ] **Step 1: Open `index.html` locally and verify the layout.**
- [ ] **Step 2: Check responsiveness by resizing the window.**
- [ ] **Step 3: Ensure Slimegrinder and Slate matches both show the new data.**
- [ ] **Step 4: Push to master.**
```bash
git push origin master
```
