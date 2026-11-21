/* MACE project page: interactive figures (vanilla JS + SVG, no dependencies).
   Data: static/js/mace_data.js (window.MD), taken from the paper's saved results. */
(function () {
  'use strict';
  var MD = window.MD, NS = 'http://www.w3.org/2000/svg';
  var C = { sage: '#b9a27a', labels: '#e3ad3c', mace: '#4f9a45', lookup: '#9aa5b1', clay: '#c0603a', ink: '#2b2b2b', grid: '#e4e0d8',
            field: '#2f7d32', staple: '#b7791f', fruit: '#b03a5b', veg: '#1f6e9c' };
  function $(id) { return document.getElementById(id); }
  function h(tag, attrs, html) { var e = document.createElement(tag); for (var k in attrs || {}) e.setAttribute(k, attrs[k]); if (html != null) e.innerHTML = html; return e; }
  function s(tag, attrs, text) { var e = document.createElementNS(NS, tag); for (var k in attrs || {}) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; }
  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function sgn(v) { return (v >= 0 ? '+' : '−') + f1(Math.abs(v)); }

  /* tooltip */
  var tip = h('div', { class: 'mc-tip', role: 'status' }); document.body.appendChild(tip);
  function showTip(ev, html) {
    tip.innerHTML = html; tip.classList.add('on');
    var w = tip.offsetWidth, hh = tip.offsetHeight, x = ev.clientX, y = ev.clientY;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 14)) + 'px';
    tip.style.top = (y - hh - 12 < 8 ? y + 18 : y - hh - 12) + 'px';
  }
  function hideTip() { tip.classList.remove('on'); }
  function bindTip(el, fn) {
    el.addEventListener('pointermove', function (ev) { showTip(ev, fn()); });
    el.addEventListener('pointerleave', hideTip);
    el.addEventListener('click', function (ev) { showTip(ev, fn()); });
  }
  /* segmented toggle */
  function seg(host, opts, cur, onPick, label) {
    var box = h('div', { class: 'seg', role: 'group', 'aria-label': label });
    opts.forEach(function (o) {
      var b = h('button', { type: 'button', class: o[0] === cur ? 'on' : '' }, o[1]);
      b.addEventListener('click', function () {
        [].forEach.call(box.children, function (c) { c.classList.remove('on'); }); b.classList.add('on'); onPick(o[0]);
      });
      box.appendChild(b);
    });
    host.appendChild(box); return box;
  }
  function svgBox(host, w, hgt) { host.innerHTML = ''; var v = s('svg', { viewBox: '0 0 ' + w + ' ' + hgt, class: 'mc-svg', role: 'img' }); host.appendChild(v); return v; }
  function yAxis(v, x0, x1, y, lo, hi, step, fmt) {
    for (var t = lo; t <= hi + 1e-9; t += step) {
      var yy = y(t);
      v.appendChild(s('line', { x1: x0, x2: x1, y1: yy, y2: yy, stroke: C.grid, 'stroke-dasharray': t === 0 ? '' : '3 3' }));
      v.appendChild(s('text', { x: x0 - 6, y: yy + 4, 'text-anchor': 'end', class: 'ax' }, fmt ? fmt(t) : t));
    }
  }

  /* ---------- 1. Recorded query player ---------- */
  function player() {
    var host = $('mc-player'); if (!host) return;
    var state = { k: 'S', step: 0, timer: null };
    var bar = h('div', { class: 'pl-bar' }); host.appendChild(bar);
    seg(bar, [['S', 'Success case'], ['F', 'Failure case']], 'S', function (k) { state.k = k; go(0); play(); }, 'Choose case');
    var btn = h('button', { type: 'button', class: 'pl-btn', 'aria-label': 'Play or pause' }, '<i class="fas fa-pause"></i>'); bar.appendChild(btn);
    var rng = h('input', { type: 'range', min: 0, max: 4, value: 0, class: 'pl-rng', 'aria-label': 'Step' }); bar.appendChild(rng);
    var lab = h('span', { class: 'pl-lab' }); bar.appendChild(lab);
    var body = h('div', { class: 'pl-body' }); host.appendChild(body);
    var names = ['The test image', 'Step 1: caption the test image', 'Step 2: retrieve four memory entries', 'Step 3: add SAGE KB claims', 'Step 4: the frozen VLM answers'];
    function render() {
      var c = MD.cases[state.k], st = state.step;
      lab.textContent = names[st]; rng.value = st;
      var refs = c.refs.map(function (r) {
        return '<div class="ref ' + (r.true ? 'hit' : '') + '"><img src="static/images/case/' + r.img + '" alt="' + r.label + ' reference"><div><b>' + r.label + '</b><i>“' + r.cap + '”</i></div></div>';
      }).join('');
      body.innerHTML =
        '<div class="pl-q"><img src="static/images/case/' + c.q + '" alt="Test image"><div><span class="kick">Question</span><b>Which disease does this cucumber plant have?</b>' +
        '<span class="truth">True disease: <b>' + c.truth + '</b></span></div></div>' +
        '<div class="pl-step ' + (st >= 1 ? 'on' : '') + '"><span class="n">1</span><div><b class="b">Caption</b> the test image without labels: <i>“' + c.test + '”</i></div></div>' +
        '<div class="pl-step ' + (st >= 2 ? 'on' : '') + '"><span class="n">2</span><div><b class="b">Retrieve</b> the four memory entries whose captions best match (BM25). Green border: true disease.<div class="refs">' + refs + '</div></div></div>' +
        '<div class="pl-step ' + (st >= 3 ? 'on' : '') + '"><span class="n">3</span><div><b class="p">SAGE KB</b> adds a class-level claim: <i>“' + c.kb + '”</i></div></div>' +
        '<div class="pl-step ' + (st >= 4 ? 'on' : '') + '"><span class="n">4</span><div><b class="y">Frozen VLM</b> reads captions and labels, never the memory images: <i>“' + c.why + '”</i>' +
        '<div class="ans"><span class="' + (c.mace === c.truth ? 'ok' : 'bad') + '"><i class="fas ' + (c.mace === c.truth ? 'fa-check-circle' : 'fa-times-circle') + '"></i> MACE: ' + c.mace + '</span>' +
        '<span class="' + (c.sage === c.truth ? 'ok' : 'bad') + '"><i class="fas ' + (c.sage === c.truth ? 'fa-check-circle' : 'fa-times-circle') + '"></i> SAGE, KB only: ' + c.sage + '</span></div>' +
        (state.k === 'F' ? '<p class="note">Two of the four retrieved entries are Anthracnose and none is Downy Mildew, so the memory pulls the answer away from the correct SAGE prediction: a retrieval miss.</p>' :
          '<p class="note">Two retrieved entries carry the true disease and describe dry, papery edges like the test leaf; the KB text alone did not separate it from Downy Mildew.</p>') + '</div></div>';
    }
    function go(st) { state.step = st; render(); }
    function play() {
      stop(); btn.innerHTML = '<i class="fas fa-pause"></i>';
      state.timer = setInterval(function () { if (state.step >= 4) { stop(); return; } go(state.step + 1); }, 1600);
    }
    function stop() { clearInterval(state.timer); state.timer = null; btn.innerHTML = '<i class="fas fa-play"></i>'; }
    btn.addEventListener('click', function () { if (state.timer) stop(); else { if (state.step >= 4) go(0); play(); } });
    rng.addEventListener('input', function () { stop(); go(+rng.value); });
    go(0);
    var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting && state.step === 0 && !state.started) { state.started = true; play(); } }); }, { threshold: 0.4 });
    io.observe(host);
  }

  /* ---------- 2. Crop explorer (Table 2) ---------- */
  function crops() {
    var host = $('mc-crops'); if (!host) return;
    var st = { sp: 'iid', m: 'Sonnet' };
    var bar = h('div', { class: 'ctl' }); host.appendChild(bar);
    seg(bar, [['iid', 'In distribution'], ['ood', 'Held-out']], 'iid', function (v) { st.sp = v; draw(); }, 'Split');
    seg(bar, [['Sonnet', 'Sonnet'], ['Haiku', 'Haiku']], 'Sonnet', function (v) { st.m = v; draw(); }, 'Model');
    var leg = h('p', { class: 'ch-sub' }); host.appendChild(leg);
    var box = h('div'); host.appendChild(box);
    var groups = [['Field', 0, 2, C.field], ['Staple', 2, 5, C.staple], ['Fruits', 5, 8, C.fruit], ['Vegetables', 8, 10, C.veg], ['Mean', 10, 11, '#555']];
    function draw() {
      var A = MD.acc[st.sp], series = [['DINOv2 1-NN', C.lookup], ['SAGE (' + st.m + ')', C.sage], ['Labels only (' + st.m + ')', C.labels], ['MACE (' + st.m + ')', C.mace]];
      leg.innerHTML = series.map(function (x) { return '<span class="k" style="--c:' + x[1] + '">' + x[0].replace(' (' + st.m + ')', '') + '</span>'; }).join('') + ' &nbsp;<span class="muted">† Bugwood held-out collection. Hover a bar for values.</span>';
      var W = 980, H = 330, L = 44, R = 10, T = 30, B = 52, v = svgBox(box, W, H);
      var y = function (val) { return T + (H - T - B) * (1 - val / 100); };
      yAxis(v, L, W - R, y, 0, 100, 20);
      var n = 11, gw = (W - L - R) / n, bw = gw * 0.78 / series.length;
      groups.forEach(function (g) {
        var x0 = L + g[1] * gw + 2, x1 = L + g[2] * gw - 2;
        v.appendChild(s('rect', { x: x0, y: 4, width: x1 - x0, height: 18, rx: 5, fill: g[3], opacity: 0.14 }));
        v.appendChild(s('text', { x: (x0 + x1) / 2, y: 17, 'text-anchor': 'middle', class: 'gl', fill: g[3] }, g[0]));
      });
      MD.crops.concat(['Mean']).forEach(function (cr, i) {
        var gx = L + i * gw + gw * 0.11;
        series.forEach(function (se, j) {
          var val = A[se[0]][i], x = gx + j * bw;
          var r = s('rect', { x: x, y: y(val), width: bw - 1.5, height: y(0) - y(val), fill: se[1], rx: 1.5, class: 'bar' });
          var best = Math.max.apply(null, series.map(function (q) { return A[q[0]][i]; }));
          if (val === best) r.setAttribute('stroke', '#1b5e20'), r.setAttribute('stroke-width', 1.4);
          bindTip(r, function () { return '<b>' + cr + '</b> · ' + (st.sp === 'iid' ? 'in distribution' : 'held-out') + '<br>' + se[0] + ': <b>' + f1(val) + '%</b>' + (se[0].indexOf('MACE') === 0 ? '<br>vs SAGE: ' + sgn(val - A['SAGE (' + st.m + ')'][i]) + ' points' : ''); });
          v.appendChild(r);
        });
        var dag = MD.bug.indexOf(cr) >= 0 ? '†' : '';
        v.appendChild(s('text', { x: L + i * gw + gw / 2, y: H - B + 16, 'text-anchor': 'middle', class: 'xl' + (cr === 'Mean' ? ' b' : '') }, cr + dag));
        var gain = A['MACE (' + st.m + ')'][i] - A['SAGE (' + st.m + ')'][i];
        v.appendChild(s('text', { x: L + i * gw + gw / 2, y: H - B + 32, 'text-anchor': 'middle', class: 'gain ' + (gain > 0.05 ? 'up' : gain < -0.05 ? 'dn' : '') }, sgn(gain)));
      });
      v.appendChild(s('text', { x: L - 36, y: T - 8, class: 'ax' }, 'Accuracy (%)'));
      v.appendChild(s('text', { x: W - R, y: H - 4, 'text-anchor': 'end', class: 'ax' }, 'second row: MACE minus SAGE, points'));
    }
    draw();
  }

  /* ---------- 3. Evidence controls ---------- */
  function evidence() {
    var host = $('mc-evid'); if (!host) return;
    var st = { sp: 'iid' }, bar = h('div', { class: 'ctl' }); host.appendChild(bar);
    seg(bar, [['iid', 'In distribution'], ['ood', 'Held-out']], 'iid', function (v) { st.sp = v; draw(); }, 'Split');
    var box = h('div'); host.appendChild(box);
    var conds = [['shuffled', 'Shuffled labels', '#d9d4cc'], ['sage', 'No memory (SAGE)', C.sage], ['random', 'Random references', '#eadfae'],
                 ['masked', 'Captions only', '#a9d1b2'], ['labels', 'Labels only', C.labels], ['mace', 'Both (MACE)', C.mace]];
    function draw() {
      var E = MD.evid[st.sp], W = 620, H = 300, L = 44, R = 10, T = 14, B = 46, v = svgBox(box, W, H);
      var y = function (val) { return T + (H - T - B) * (1 - (val - 20) / 60); };
      yAxis(v, L, W - R, y, 20, 80, 10);
      var bw = (W - L - R) / conds.length;
      conds.forEach(function (c, i) {
        var e = E[c[0]], x = L + i * bw + bw * 0.16, w = bw * 0.68;
        var r = s('rect', { x: x, y: y(e[0]), width: w, height: y(20) - y(e[0]), fill: c[2], rx: 3, class: 'bar grow' });
        bindTip(r, function () { return '<b>' + c[1] + '</b><br>' + f1(e[0]) + '% [' + f1(e[1]) + ', ' + f1(e[2]) + ']<br>vs MACE: ' + sgn(e[0] - E.mace[0]) + ' points'; });
        v.appendChild(r);
        v.appendChild(s('line', { x1: x + w / 2, x2: x + w / 2, y1: y(e[1]), y2: y(e[2]), stroke: '#333', 'stroke-width': 1.2 }));
        v.appendChild(s('text', { x: x + w / 2, y: y(e[2]) - 5, 'text-anchor': 'middle', class: 'val' }, f1(e[0])));
        var words = c[1].split(' ');
        v.appendChild(s('text', { x: x + w / 2, y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, words.slice(0, 2).join(' ')));
        if (words.length > 2) v.appendChild(s('text', { x: x + w / 2, y: H - B + 30, 'text-anchor': 'middle', class: 'xl' }, words.slice(2).join(' ')));
      });
    }
    draw();
  }

  /* ---------- 4. Single answers ---------- */
  function answers() {
    var host = $('mc-ans'); if (!host) return;
    var st = { m: 'sonnet', sp: 'iid' }, bar = h('div', { class: 'ctl' }); host.appendChild(bar);
    seg(bar, [['sonnet', 'Sonnet'], ['haiku', 'Haiku']], 'sonnet', function (v) { st.m = v; draw(); }, 'Model');
    seg(bar, [['iid', 'In distribution'], ['ood', 'Held-out']], 'iid', function (v) { st.sp = v; draw(); }, 'Split');
    var grid = h('div', { class: 'two' }); host.appendChild(grid);
    var a = h('div'), b = h('div'); grid.appendChild(a); grid.appendChild(b);
    function draw() {
      var key = st.m + '|' + st.sp, fx = MD.fix[key], g = MD.grad[key];
      a.innerHTML = '<div class="big2"><div><span class="v up">' + f1(fx[0]) + '%</span><span class="l">SAGE errors MACE fixes</span></div><div><span class="v dn">' + f1(fx[1]) + '%</span><span class="l">new errors MACE introduces</span></div></div>' +
        '<p class="muted">Share of queries; exact McNemar p &lt; 0.001 for both models on both splits.</p>';
      var W = 460, H = 250, L = 40, R = 10, T = 14, B = 40, v = svgBox(b, W, H);
      var x = function (n) { return L + (W - L - R) * n / 4; }, y = function (val) { return T + (H - T - B) * (1 - val / 100); };
      yAxis(v, L, W - R, y, 0, 100, 25);
      [['mace', 0, C.mace, 'MACE'], ['sage', 1, C.sage, 'SAGE']].forEach(function (se) {
        var d = [0, 1, 2, 3, 4].map(function (n) { return [x(n), y(g[n][se[1]])]; });
        v.appendChild(s('polyline', { points: d.map(function (p) { return p.join(','); }).join(' '), fill: 'none', stroke: se[2], 'stroke-width': 2.6 }));
        d.forEach(function (p, n) {
          var c = s('circle', { cx: p[0], cy: p[1], r: 5, fill: se[2], stroke: '#fff', 'stroke-width': 1.5, class: 'dot' });
          bindTip(c, function () { return '<b>' + n + ' of 4</b> references carry the true disease<br>' + se[3] + ': ' + f1(g[n][se[1]]) + '% (n = ' + g[n][2] + ')'; });
          v.appendChild(c);
        });
      });
      [0, 1, 2, 3, 4].forEach(function (n) { v.appendChild(s('text', { x: x(n), y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, n)); });
      v.appendChild(s('text', { x: (W + L) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ax' }, 'True-disease references among the four'));
      v.appendChild(s('text', { x: W - R, y: y(g[4][0]) + 18, 'text-anchor': 'end', class: 'tag', fill: C.mace }, 'MACE'));
      v.appendChild(s('text', { x: W - R, y: y(g[4][1]) + 18, 'text-anchor': 'end', class: 'tag', fill: '#8a7550' }, 'SAGE'));
    }
    draw();
  }

  /* ---------- 5. Memory: source and coverage ---------- */
  function memory() {
    var host = $('mc-mem'); if (!host) return;
    var st = { m: 'sonnet' }, bar = h('div', { class: 'ctl' }); host.appendChild(bar);
    seg(bar, [['sonnet', 'Sonnet'], ['haiku', 'Haiku']], 'sonnet', function (v) { st.m = v; draw(); }, 'Model');
    var grid = h('div', { class: 'two' }); host.appendChild(grid);
    var a = h('div'), b = h('div'); grid.appendChild(a); grid.appendChild(b);
    (function src() {
      var W = 470, H = 250, L = 40, R = 10, T = 14, B = 40, v = svgBox(a, W, H), y = function (val) { return T + (H - T - B) * (1 - (val - 20) / 70); };
      yAxis(v, L, W - R, y, 20, 90, 10);
      [['sage', 'SAGE', C.sage], ['other_source', 'Other collections', C.labels], ['same_source', 'Same collection', C.mace]].forEach(function (c, i) {
        var e = MD.src[c[0]], bw = (W - L - R) / 3, x = L + i * bw + bw * 0.18, w = bw * 0.64;
        var r = s('rect', { x: x, y: y(e[0]), width: w, height: y(20) - y(e[0]), fill: c[2], rx: 3, class: 'bar grow' });
        bindTip(r, function () { return '<b>' + c[1] + '</b> (Sonnet, IID)<br>' + f1(e[0]) + '% [' + f1(e[1]) + ', ' + f1(e[2]) + ']'; });
        v.appendChild(r);
        v.appendChild(s('line', { x1: x + w / 2, x2: x + w / 2, y1: y(e[1]), y2: y(e[2]), stroke: '#333', 'stroke-width': 1.2 }));
        v.appendChild(s('text', { x: x + w / 2, y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, c[1].split(' ')[0]));
      });
      v.appendChild(s('text', { x: (W + L) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ax' }, 'Memory from (Sonnet, IID)'));
    })();
    function draw() {
      var W = 470, H = 250, L = 40, R = 10, T = 14, B = 40, v = svgBox(b, W, H);
      var x = function (val) { return L + (W - L - R) * (val - 30) / 72; }, y = function (val) { return T + (H - T - B) * (1 - (val + 15) / 65); };
      yAxis(v, L, W - R, y, -10, 50, 10);
      var idx = st.m === 'sonnet' ? 3 : 4, pts = MD.cov.map(function (r) { return [r[2], r[idx], r]; });
      var n = pts.length, mx = pts.reduce(function (a, p) { return a + p[0]; }, 0) / n, my = pts.reduce(function (a, p) { return a + p[1]; }, 0) / n;
      var sxy = 0, sxx = 0; pts.forEach(function (p) { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) * (p[0] - mx); });
      var k = sxy / sxx; v.appendChild(s('line', { x1: x(33), x2: x(100), y1: y(my + k * (33 - mx)), y2: y(my + k * (100 - mx)), stroke: '#888', 'stroke-dasharray': '5 4' }));
      pts.forEach(function (p) {
        var bug = MD.bug.indexOf(p[2][0]) >= 0 && p[2][1] === 'ood', col = p[2][1] === 'iid' ? C.mace : C.clay;
        var c = s(p[2][1] === 'iid' ? 'circle' : 'rect', p[2][1] === 'iid' ? { cx: x(p[0]), cy: y(p[1]), r: 6 } : { x: x(p[0]) - 5.5, y: y(p[1]) - 5.5, width: 11, height: 11, transform: 'rotate(45 ' + x(p[0]) + ' ' + y(p[1]) + ')' });
        c.setAttribute('fill', col); c.setAttribute('stroke', '#fff'); c.setAttribute('stroke-width', 1.4); c.setAttribute('class', 'dot');
        bindTip(c, function () { return '<b>' + p[2][0] + '</b>, ' + (p[2][1] === 'iid' ? 'in distribution' : 'held-out' + (bug ? ' (Bugwood)' : '')) + '<br>true disease retrieved: ' + f1(p[0]) + '%<br>gain over SAGE: ' + sgn(p[1]) + ' points'; });
        v.appendChild(c);
      });
      v.appendChild(s('text', { x: (W + L) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ax' }, 'True disease retrieved among the four (%)'));
      v.appendChild(s('text', { x: 4, y: T - 2, class: 'ax' }, 'Gain over SAGE (points)'));
      v.appendChild(s('text', { x: L + 6, y: T + 12, class: 'tag' }, 'r = ' + (st.m === 'sonnet' ? '0.49 (perm. p = 0.03)' : '0.64 (perm. p = 0.004)')));
      [40, 60, 80, 100].forEach(function (t) { v.appendChild(s('text', { x: x(t), y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, t)); });
    }
    draw();
  }

  /* ---------- 6. Selection rules ---------- */
  function selection() {
    var host = $('mc-sel'); if (!host) return;
    var grid = h('div', { class: 'two' }); host.appendChild(grid);
    var a = h('div'), b = h('div'); grid.appendChild(a); grid.appendChild(b);
    (function cov() {
      var W = 420, H = 250, L = 40, R = 10, T = 14, B = 40, v = svgBox(a, W, H);
      var x = function (k) { return L + (W - L - R) * (k - 1) / 19; }, y = function (val) { return T + (H - T - B) * (1 - (val - 30) / 72); };
      yAxis(v, L, W - R, y, 30, 100, 10);
      v.appendChild(s('rect', { x: x(1) - 4, y: T, width: x(4) - x(1) + 8, height: H - T - B, fill: '#eef3e6' }));
      v.appendChild(s('text', { x: (x(1) + x(4)) / 2, y: H - B - 8, 'text-anchor': 'middle', class: 'tag', fill: '#2f6b2a' }, 'to VLM'));
      [['iid', C.mace, 'IID'], ['ood', C.clay, 'Held-out']].forEach(function (se) {
        var d = MD.hitk[se[0]];
        v.appendChild(s('polyline', { points: d.map(function (val, i) { return x(i + 1) + ',' + y(val); }).join(' '), fill: 'none', stroke: se[1], 'stroke-width': 2.4 }));
        d.forEach(function (val, i) {
          var c = s('circle', { cx: x(i + 1), cy: y(val), r: 6, fill: se[1], opacity: 0.0, class: 'dot' });
          bindTip(c, function () { return '<b>k = ' + (i + 1) + '</b>, ' + se[2] + '<br>true disease among BM25 top ' + (i + 1) + ': ' + f1(val) + '%'; });
          v.appendChild(c);
        });
        v.appendChild(s('text', { x: x(20) - 2, y: y(d[19]) + (se[0] === 'iid' ? -8 : 16), 'text-anchor': 'end', class: 'tag', fill: se[1] }, se[2]));
      });
      [1, 4, 10, 20].forEach(function (t) { v.appendChild(s('text', { x: x(t), y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, t)); });
      v.appendChild(s('text', { x: (W + L) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ax' }, 'BM25 candidates retrieved (k)'));
    })();
    var st = { m: 'sonnet' }, bar = h('div', { class: 'ctl' }); b.appendChild(bar);
    seg(bar, [['sonnet', 'Sonnet'], ['haiku', 'Haiku']], 'sonnet', function (v) { st.m = v; draw(); }, 'Model');
    var box = h('div'); b.appendChild(box);
    function draw() {
      var W = 420, H = 222, L = 40, R = 10, T = 14, B = 40, v = svgBox(box, W, H);
      var y = function (val) { return T + (H - T - B) * (1 - (val + 3) / 8); };
      yAxis(v, L, W - R, y, -3, 5, 1, function (t) { return t > 0 ? '+' + t : t; });
      var rules = [['LLM-written MLP', C.mace], ['BM25 + DINOv2', C.labels]], splits = [['iid', 'In distribution'], ['ood', 'Held-out']];
      var gw = (W - L - R) / 2, bw = gw * 0.3;
      splits.forEach(function (sp, i) {
        rules.forEach(function (ru, j) {
          var p = MD.sel.filter(function (q) { return q[0] === ru[0] && q[1] === st.m && q[2] === sp[0]; })[0];
          var x = L + i * gw + gw * 0.2 + j * bw * 1.1, top = Math.min(y(p[4]), y(0)), hh = Math.abs(y(p[4]) - y(0));
          var r = s('rect', { x: x, y: top, width: bw, height: Math.max(hh, 1), fill: ru[1], rx: 2, class: 'bar' });
          bindTip(r, function () { return '<b>' + ru[0] + '</b>, ' + (st.m === 'sonnet' ? 'Sonnet' : 'Haiku') + ', ' + sp[1] + '<br>accuracy vs BM25: ' + sgn(p[4]) + ' [' + sgn(p[5][0]) + ', ' + sgn(p[5][1]) + ']<br>hit@4 vs BM25: ' + sgn(p[3]) + '<br>not significant'; });
          v.appendChild(r);
          v.appendChild(s('line', { x1: x + bw / 2, x2: x + bw / 2, y1: y(Math.max(-3, p[5][0])), y2: y(Math.min(5, p[5][1])), stroke: '#333', 'stroke-width': 1.1 }));
          v.appendChild(s('text', { x: x + bw / 2, y: (p[4] >= 0 ? y(p[4]) : y(0)) - 5, 'text-anchor': 'middle', class: 'val' }, sgn(p[4])));
        });
        v.appendChild(s('text', { x: L + i * gw + gw / 2, y: H - B + 16, 'text-anchor': 'middle', class: 'xl' }, sp[1]));
      });
      v.appendChild(s('text', { x: (W + L) / 2, y: H - 6, 'text-anchor': 'middle', class: 'ax' }, 'Accuracy change from BM25 (points), 95% intervals'));
    }
    var leg = h('p', { class: 'ch-sub' }, '<span class="k" style="--c:' + C.mace + '">LLM-written MLP</span><span class="k" style="--c:' + C.labels + '">BM25 + DINOv2</span>');
    b.appendChild(leg); draw();
  }

  /* ---------- paper figure lightbox ---------- */
  function lightbox() {
    var lb = h('div', { class: 'mc-lb', role: 'dialog', 'aria-label': 'Figure as printed in the paper' }, '<img alt=""><span class="x">&times;</span>');
    document.body.appendChild(lb);
    lb.addEventListener('click', function () { lb.classList.remove('on'); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.classList.remove('on'); });
    [].forEach.call(document.querySelectorAll('.pf-btn'), function (btn) {
      btn.addEventListener('click', function () { lb.querySelector('img').src = btn.getAttribute('data-src'); lb.classList.add('on'); });
    });
  }

  function init() { player(); crops(); evidence(); answers(); memory(); selection(); lightbox(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
