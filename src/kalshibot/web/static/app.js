// Shared Chart.js configuration for consistent styling.
(function () {
  if (window.Chart) {
    const css = getComputedStyle(document.documentElement);
    const text   = css.getPropertyValue('--text').trim();
    const dim    = css.getPropertyValue('--text-dim').trim();
    const border = css.getPropertyValue('--border').trim();
    Chart.defaults.color = dim || '#9aa7c4';
    Chart.defaults.font.family = 'ui-sans-serif, -apple-system, system-ui, Segoe UI, Roboto';
    Chart.defaults.font.size = 12;
    Chart.defaults.borderColor = border || '#1f2a44';
    Chart.defaults.plugins.legend.labels.color = text || '#e6ecff';
    Chart.defaults.scale.grid = { color: 'rgba(148,163,184,.08)' };
    Chart.defaults.scale.ticks = { color: dim || '#9aa7c4' };
    Chart.defaults.plugins.tooltip.backgroundColor = '#0b1020';
    Chart.defaults.plugins.tooltip.borderColor = border || '#1f2a44';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = text || '#e6ecff';
    Chart.defaults.plugins.tooltip.bodyColor = dim || '#9aa7c4';
  }
})();

window.KB = {
  palette: ['#7dd3fc','#c4b5fd','#fca5a5','#86efac','#fcd34d','#fdba74',
            '#f9a8d4','#67e8f9','#bef264','#f0abfc','#a5f3fc','#fde68a'],
  fmt: {
    money: (v) => (v >= 0 ? '+' : '') + Number(v).toFixed(4),
    pct:   (v) => (v * 100).toFixed(2) + '%',
    num2:  (v) => Number(v).toFixed(2),
  },
  lineDataset: (label, s, color, idx = 0) => ({
    label,
    data: s.y,
    borderColor: color,
    backgroundColor: color + '33',
    pointRadius: 0,
    tension: 0.15,
    borderWidth: 1.6,
    fill: idx === 0 ? true : false,
  }),
  async fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('fetch ' + url + ' failed');
    return res.json();
  },
  /* Smoothly count an element's text from prev to next, ~500ms.
     Auto-detects $, ¢, %, and decimal places to keep formatting. */
  countTo(el, next, formatter) {
    if (!el) return;
    const prev = parseFloat(el.dataset.kbValue || '0');
    el.dataset.kbValue = String(next);
    if (Math.abs(prev - next) < 0.0005) {
      el.textContent = formatter ? formatter(next) : next;
      return;
    }
    const start = performance.now();
    const dur = 480;
    function step(now) {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);   // easeOutCubic
      const val = prev + (next - prev) * eased;
      el.textContent = formatter ? formatter(val) : val.toFixed(2);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  },
  /* Add flash class briefly so the user notices a value updated. */
  flashChange(el, dir) {
    if (!el) return;
    el.classList.remove('flash-up', 'flash-down');
    void el.offsetWidth;   // restart animation
    el.classList.add(dir > 0 ? 'flash-up' : 'flash-down');
  },
  liveDot() { return '<span class="live-dot"></span>'; },
};
