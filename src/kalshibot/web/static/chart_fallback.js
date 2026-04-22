// Minimal Chart.js shim. If the real Chart.js fails to load (slow network,
// blocked CDN, whatever), this renders passable line / bar charts onto
// any <canvas> so the dashboard doesn't come up blank.
(function () {
  function installed() {
    return typeof window.Chart === 'function' && !window.__chartCdnFailed;
  }

  // Wait a beat for the CDN to resolve, then decide.
  setTimeout(() => {
    if (installed()) return;
    window.__chartCdnFailed = true;
    window.Chart = FakeChart;
    console.warn('[kalshibot] Chart.js CDN not available, using built-in renderer.');
  }, 800);

  function FakeChart(canvas, cfg) {
    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;
    const draw = () => render(ctx, canvas, cfg, parent);
    draw();
    window.addEventListener('resize', draw, { passive: true });
    return { destroy() {}, update: draw };
  }

  function render(ctx, canvas, cfg, parent) {
    // HiDPI-aware sizing
    const dpr = window.devicePixelRatio || 1;
    const W = (parent && parent.clientWidth) || 600;
    const H = (parent && parent.clientHeight) || 240;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // Background
    ctx.fillStyle = '#111a32';
    ctx.fillRect(0, 0, W, H);

    const pad = { l: 40, r: 14, t: 16, b: 24 };
    const inner = { x: pad.l, y: pad.t, w: W - pad.l - pad.r, h: H - pad.t - pad.b };
    const type = cfg.type || 'line';
    const datasets = (cfg.data && cfg.data.datasets) || [];
    const labels   = (cfg.data && cfg.data.labels) || [];

    // Figure out bounds
    let min = Infinity, max = -Infinity;
    datasets.forEach(d => (d.data || []).forEach(v => {
      const y = typeof v === 'object' ? v.y : v;
      if (y == null) return;
      if (y < min) min = y;
      if (y > max) max = y;
    }));
    if (!isFinite(min) || !isFinite(max) || min === max) { min = 0; max = 1; }
    const pad2 = (max - min) * 0.1;
    min -= pad2; max += pad2;

    // Grid + y axis
    ctx.strokeStyle = 'rgba(148,163,184,.1)';
    ctx.fillStyle   = '#6b7897';
    ctx.font        = '11px ui-sans-serif';
    ctx.textAlign   = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const y = inner.y + inner.h - (i / 4) * inner.h;
      ctx.beginPath(); ctx.moveTo(inner.x, y); ctx.lineTo(inner.x + inner.w, y); ctx.stroke();
      const val = min + (i / 4) * (max - min);
      ctx.fillText(val.toFixed(2), inner.x - 6, y);
    }

    const palette = ['#7dd3fc', '#c4b5fd', '#fca5a5', '#86efac', '#fcd34d', '#fdba74'];
    const toY = v => inner.y + inner.h - ((v - min) / (max - min)) * inner.h;

    if (type === 'line') {
      datasets.forEach((d, i) => {
        const color = d.borderColor || palette[i % palette.length];
        const vals = d.data || [];
        if (!vals.length) return;
        const step = inner.w / Math.max(1, vals.length - 1);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        vals.forEach((v, k) => {
          const y = typeof v === 'object' ? v.y : v;
          const xv = inner.x + k * step;
          if (k === 0) ctx.moveTo(xv, toY(y));
          else ctx.lineTo(xv, toY(y));
        });
        ctx.stroke();
      });
    } else if (type === 'bar') {
      const groups = labels.length || (datasets[0] && datasets[0].data.length) || 0;
      const gw = inner.w / Math.max(1, groups);
      datasets.forEach((d, i) => {
        const color = (typeof d.backgroundColor === 'string')
          ? d.backgroundColor : palette[i % palette.length];
        (d.data || []).forEach((v, k) => {
          const y = typeof v === 'object' ? v.y : v;
          const barX = inner.x + k * gw + gw * 0.15;
          const barW = gw * 0.7;
          const top = toY(Math.max(y, 0));
          const bot = toY(Math.min(y, 0));
          ctx.fillStyle = color;
          ctx.fillRect(barX, Math.min(top, bot), barW, Math.abs(bot - top) || 1);
        });
      });
      // X labels (sparse)
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillStyle = '#6b7897';
      const every = Math.max(1, Math.ceil(labels.length / 8));
      labels.forEach((lab, k) => {
        if (k % every !== 0) return;
        ctx.fillText(String(lab).slice(0, 10),
                     inner.x + k * gw + gw / 2, inner.y + inner.h + 4);
      });
    } else if (type === 'doughnut') {
      const cx = inner.x + inner.w / 2;
      const cy = inner.y + inner.h / 2;
      const r  = Math.min(inner.w, inner.h) / 2 - 10;
      const vals = (datasets[0] && datasets[0].data) || [];
      const total = vals.reduce((a, b) => a + (b || 0), 0) || 1;
      let start = -Math.PI / 2;
      vals.forEach((v, k) => {
        const ang = (v / total) * Math.PI * 2;
        ctx.fillStyle = palette[k % palette.length];
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, start, start + ang);
        ctx.closePath(); ctx.fill();
        start += ang;
      });
      ctx.fillStyle = '#111a32';
      ctx.beginPath(); ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2); ctx.fill();
    } else if (type === 'scatter') {
      datasets.forEach((d, i) => {
        const color = d.borderColor || d.backgroundColor || palette[i % palette.length];
        ctx.fillStyle = color;
        (d.data || []).forEach(p => {
          const yy = toY(p.y);
          const xx = inner.x + (p.x - (min)) / ((max) - (min)) * inner.w; // rough
          ctx.beginPath(); ctx.arc(xx, yy, 3, 0, Math.PI * 2); ctx.fill();
        });
      });
    }
  }
})();
