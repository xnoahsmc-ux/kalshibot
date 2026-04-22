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
};
