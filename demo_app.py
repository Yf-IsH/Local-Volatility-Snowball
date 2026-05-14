import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dupire import LocalVolPricer
from core.surface_factory import build_demo_iv_surface
from data.market_data import INDEX_UNIVERSE, load_market_snapshot
from data.sse_option_iv import ETF_OPTION_UNIVERSE, load_sse_option_iv_surface
from pricer.pde_pricer import PDEPricer
from pricer.snowball_mc import SnowballMCPricer, SnowballTerms


def surface_payload(surface, lv):
    strike_ratios = np.linspace(0.7, 1.35, 16)
    if len(surface.T_grid) <= 7:
        maturities = surface.T_grid
    else:
        maturities = np.linspace(surface.min_maturity, surface.max_maturity, 7)
    iv_rows = []
    local_rows = []
    for T in maturities:
        iv_row = []
        local_row = []
        for ratio in strike_ratios:
            strike = lv.S * ratio
            iv_row.append(surface.get_iv(strike, T))
            local_row.append(lv.local_vol(strike, T))
        iv_rows.append(iv_row)
        local_rows.append(local_row)
    return {
        "strike_ratios": strike_ratios.round(4).tolist(),
        "maturities": maturities.round(4).tolist(),
        "iv": np.asarray(iv_rows).round(6).tolist(),
        "local_vol": np.asarray(local_rows).round(6).tolist(),
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Vol Snowball Coupon Demo</title>
  <style>
    :root { color-scheme: light; --ink: #172026; --muted: #66727c; --line: #d9e0e6; --fill: #f5f7f9; --brand: #116a7b; --soft: #eef6f7; --warn: #c75146; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color: var(--ink); background: #fff; }
    header { padding: 22px 28px 16px; border-bottom: 1px solid var(--line); display: flex; align-items: end; justify-content: space-between; gap: 18px; }
    h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 6px; }
    main { padding: 18px 28px 28px; display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 20px; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .controls { padding: 16px; display: grid; gap: 14px; align-content: start; }
    label { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
    select, input { width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; font-size: 14px; color: var(--ink); background: #fff; }
    button { border: 0; border-radius: 6px; padding: 10px 12px; background: var(--brand); color: #fff; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .workspace { display: grid; gap: 14px; min-width: 0; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 84px; }
    .metric.primary { background: var(--soft); border-color: #b9d7db; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; font-size: 21px; margin-top: 7px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .metric.primary strong { font-size: 27px; color: var(--brand); }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .chart-wrap, .surface-wrap { min-height: 380px; padding: 12px; }
    canvas { width: 100%; max-height: 360px; }
    .surface-wrap canvas { cursor: grab; max-height: 420px; }
    .surface-wrap canvas:active { cursor: grabbing; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-top: 1px solid var(--line); text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .explain { padding: 16px; line-height: 1.65; font-size: 13px; color: #2c383f; }
    .flow { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
    .flow span { border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; background: var(--fill); }
    code { background: var(--fill); border: 1px solid var(--line); border-radius: 5px; padding: 2px 5px; font-family: Consolas, monospace; }
    .formula-toggle { margin-top: 12px; border: 1px solid var(--brand); background: #fff; color: var(--brand); }
    .derivation { display: none; margin-top: 14px; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }
    .derivation.open { display: block; }
    .math { overflow-x: auto; padding: 10px 12px; margin: 8px 0; border: 1px solid var(--line); border-radius: 6px; background: #fff; font-family: Cambria Math, "Times New Roman", serif; font-size: 15px; white-space: nowrap; }
    .derivation p { margin: 8px 0; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.6; padding: 12px 16px; background: var(--fill); border-top: 1px solid var(--line); }
    .status { font-size: 13px; color: var(--muted); min-height: 20px; }
    @media (max-width: 1080px) { .chart-grid { grid-template-columns: 1fr; } }
    @media (max-width: 920px) { main { grid-template-columns: 1fr; padding: 14px; } header { padding: 18px 14px 12px; display: block; } .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Local Vol Snowball Coupon Demo</h1>
      <div class="sub">用波动率曲面校准局部波动率，再反解雪球公允年化票息</div>
    </div>
    <div class="status" id="status"></div>
  </header>
  <main>
    <section class="panel controls">
      <label>标的指数
        <select id="index">
          <optgroup label="真实IV ETF期权">
            <option value="etf50">50ETF</option>
            <option value="etf300">300ETF</option>
            <option value="etf500">500ETF</option>
            <option value="kc50">科创50ETF</option>
          </optgroup>
          <optgroup label="指数演示曲面">
          <option value="csi500">中证500</option>
          <option value="csi1000">中证1000</option>
          <option value="csi300">沪深300</option>
          <option value="sse50">上证50</option>
          </optgroup>
        </select>
      </label>
      <label>名义本金
        <input id="notional" type="number" value="1000000" step="10000">
      </label>
      <label>期限 年
        <input id="maturity" type="number" value="1" step="0.25" min="0.25" max="2">
      </label>
      <label>手动报价票息 年化
        <input id="coupon" type="number" value="0.16" step="0.01" min="0" max="1">
      </label>
      <label>敲出水平 S0倍数
        <input id="ko" type="number" value="1.03" step="0.01" min="0.8" max="1.5">
      </label>
      <label>敲入水平 S0倍数
        <input id="ki" type="number" value="0.75" step="0.01" min="0.3" max="1">
      </label>
      <label>模拟路径
        <input id="paths" type="number" value="5000" step="1000" min="1000" max="30000">
      </label>
      <button id="run">计算公允票息</button>
    </section>
    <section class="workspace">
      <div class="grid" id="metrics"></div>
      <section class="panel explain">
        <h2>模型逻辑</h2>
        <div class="flow">
          <span>市场 IV(K,T)</span><span>Dupire</span><span>σlocal(S,t)</span><span>模拟雪球路径</span><span>反解 coupon</span>
        </div>
        <div>Local volatility 假设 <code>dS/S = (r-q)dt + σlocal(S,t)dW</code>。市场上每个行权价 <code>K</code>、到期日 <code>T</code> 的期权都有一个隐含波动率 <code>IV(K,T)</code>，这张曲面代表市场对未来分布、偏斜和期限结构的定价。</div>
        <div>Dupire 公式把 vanilla option 的价格曲面转换成 <code>σlocal(S,t)</code>。有了这个状态依赖波动率后，模型既能大致复现 vanilla 期权曲面，也能模拟雪球这种路径依赖产品的敲入、敲出和到期现金流。</div>
        <div>券商报价常见输出不是“雪球值多少钱”，而是让 <code>PV(coupon) = notional</code> 成立的年化票息。左侧“手动报价票息”只是给你输入一个假设报价，用来对比这个票息下 PV 是高于还是低于本金；真正由模型反解出来的是上方的“公允年化票息”。</div>
        <div>概率按最终路径分成四类，且总和应接近 100%：未敲入未敲出、未敲入后敲出、敲入未敲出、敲入后敲出。本页的“红利概率”指未敲入未敲出，也就是持有到期并获得票息、没有本金亏损的路径占比。</div>
        <div>ETF 标的会优先使用上交所真实期权风险指标里的 <code>IMPLC_VOLATLTY</code> 构建 IV 曲面；指数标的暂时仍使用历史波动率演示曲面。</div>
        <button class="formula-toggle" id="formulaToggle" type="button">查看详细数学推导</button>
        <div class="derivation" id="derivation">
          <p><strong>1. 风险中性标的过程</strong></p>
          <div class="math">dS_t / S_t = (r - q) dt + σ_loc(S_t,t) dW_t^Q</div>
          <p>其中 <code>r</code> 是无风险利率，<code>q</code> 是连续分红率或持有成本近似，<code>σ_loc(S,t)</code> 是局部波动率。它不是一个常数，而是随价格位置和时间变化的函数。</p>
          <p><strong>2. 从 IV 曲面到期权价格曲面</strong></p>
          <div class="math">C(K,T) = BSCall(S0, K, T, r, q, IV(K,T))</div>
          <p>交易所或市场报价给出不同执行价和到期日的隐含波动率 <code>IV(K,T)</code>。先把它转换成欧式看涨期权价格曲面 <code>C(K,T)</code>。</p>
          <p><strong>3. Dupire 局部波动率公式</strong></p>
          <div class="math">σ_loc²(K,T) = [∂C/∂T + qC + (r-q)K∂C/∂K] / [0.5 K² ∂²C/∂K²]</div>
          <p>这一步把 vanilla option 的市场曲面转换成 local volatility surface。直觉上，期权价格随期限的变化提供“时间方向”的信息，随行权价的一阶和二阶变化提供“分布形状/密度”的信息。</p>
          <p><strong>4. 用 local vol 模拟雪球路径</strong></p>
          <div class="math">S_{t+Δt} = S_t exp((r-q-0.5σ_loc²)Δt + σ_loc sqrt(Δt)Z)</div>
          <p>每条路径按每日步长演化，并检查敲入线、观察日敲出线和到期现金流。</p>
          <p><strong>5. 公允票息反解</strong></p>
          <div class="math">PV(c) = E[Dτ · PrincipalPayoff] + c · E[Dτ · Accrual]</div>
          <div class="math">c_fair = (Notional - PV_principal) / PV_coupon_annuity</div>
          <p>所以票息不是模型直接“猜”的，而是在同一批路径上拆成本金腿和票息年金腿，再令理论价值等于名义本金反解出来。</p>
        </div>
      </section>
      <section class="panel chart-wrap">
        <h2>指数收盘走势</h2>
        <canvas id="chart"></canvas>
      </section>
      <section class="chart-grid">
        <section class="panel surface-wrap">
          <h2>输入隐含波动率曲面 IV(K,T)</h2>
          <canvas id="ivSurface"></canvas>
        </section>
        <section class="panel surface-wrap">
          <h2>Dupire 局部波动率曲面 σlocal(S,t)</h2>
          <canvas id="lvSurface"></canvas>
        </section>
      </section>
      <section class="panel">
        <table>
          <thead><tr><th>项目</th><th>数值</th></tr></thead>
          <tbody id="details"></tbody>
        </table>
        <div class="note">ETF 期权 IV 来自上交所期权风险指标 IMPLC_VOLATLTY，并会过滤 0 或极端 IV 后插值成规则曲面。指数标的没有直接接入股指期权链，因此仍显示演示曲面。</div>
      </section>
    </section>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    let historyData = null;
    let surfaceData = null;
    const surfaceViews = {
      ivSurface: { yaw: -0.72, pitch: 0.58, dragging: false, hover: null },
      lvSurface: { yaw: -0.72, pitch: 0.58, dragging: false, hover: null },
    };

    function fmtMoney(x) { return Number(x).toLocaleString('zh-CN', { maximumFractionDigits: 0 }); }
    function fmtPct(x) { return (Number(x) * 100).toFixed(2) + '%'; }
    function metric(label, value, primary=false) { return `<div class="metric ${primary ? 'primary' : ''}"><span>${label}</span><strong>${value}</strong></div>`; }

    async function run() {
      $('run').disabled = true;
      $('status').textContent = '定价中...';
      const q = new URLSearchParams({
        index: $('index').value,
        notional: $('notional').value,
        maturity: $('maturity').value,
        coupon: $('coupon').value,
        ko: $('ko').value,
        ki: $('ki').value,
        paths: $('paths').value,
      });
      try {
        const res = await fetch('/api/price?' + q.toString());
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'request failed');
        $('metrics').innerHTML = [
          metric('公允年化票息', fmtPct(data.fair_coupon), true),
          metric('手动票息PV', fmtMoney(data.quoted_price)),
          metric('标的现货', data.spot.toFixed(2)),
          metric('数据源', data.source),
        ].join('');
        $('details').innerHTML = [
          ['指数', `${data.name} (${data.code})`],
          ['行情日期', data.asof],
          ['IV曲面来源', data.iv_source],
          ['IV曲面日期', data.iv_asof],
          ['IV有效点数', data.iv_points],
          ['历史年化波动率', fmtPct(data.annual_vol)],
          ['手动报价票息', fmtPct(data.quoted_coupon)],
          ['公允票息下PV', fmtMoney(data.par_price)],
          ['本金腿PV', fmtMoney(data.principal_pv)],
          ['票息年金腿PV 每100%票息', fmtMoney(data.coupon_annuity_pv)],
          ['欧式ATM Call校验价', data.european_call.toFixed(4)],
          ['ATM局部波动率', fmtPct(data.atm_local_vol)],
          ['红利概率：未敲入未敲出', fmtPct(data.bonus_probability)],
          ['未敲入后敲出', fmtPct(data.no_ki_ko_probability)],
          ['敲入未敲出', fmtPct(data.ki_no_ko_probability)],
          ['敲入后敲出', fmtPct(data.ki_ko_probability)],
          ['四类概率合计', fmtPct(data.probability_total)],
          ['曾敲出概率', fmtPct(data.ko_probability)],
          ['曾敲入概率', fmtPct(data.ki_probability)],
          ['平均敲出时间', data.avg_ko_time ? data.avg_ko_time.toFixed(3) + ' 年' : '-'],
          ['蒙特卡洛标准误', fmtMoney(data.std_error)],
        ].map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

        historyData = data.history;
        surfaceData = data.surface;
        drawAll();
        $('status').textContent = '完成';
      } catch (err) {
        $('status').textContent = err.message;
      } finally {
        $('run').disabled = false;
      }
    }

    $('run').addEventListener('click', run);
    $('formulaToggle').addEventListener('click', () => {
      const panel = $('derivation');
      panel.classList.toggle('open');
      $('formulaToggle').textContent = panel.classList.contains('open') ? '收起数学推导' : '查看详细数学推导';
    });
    window.addEventListener('resize', drawAll);
    for (const id of Object.keys(surfaceViews)) bindSurfaceInteraction(id);

    function drawAll() {
      drawChart();
      drawSurface('ivSurface', surfaceData && surfaceData.iv);
      drawSurface('lvSurface', surfaceData && surfaceData.local_vol);
    }

    function setupCanvas(id, height) {
      const canvas = $(id);
      const parent = canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(parent.clientWidth - 24, 280);
      canvas.width = w * dpr;
      canvas.height = height * dpr;
      canvas.style.height = height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, height);
      return { canvas, ctx, w, h: height };
    }

    function drawChart() {
      if (!historyData) return;
      const { ctx, w, h } = setupCanvas('chart', 330);
      const values = historyData.close.map(Number);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const pad = { left: 54, right: 16, top: 18, bottom: 34 };
      const iw = w - pad.left - pad.right;
      const ih = h - pad.top - pad.bottom;
      const y = v => pad.top + (max - v) / Math.max(max - min, 1e-9) * ih;
      const x = i => pad.left + i / Math.max(values.length - 1, 1) * iw;
      ctx.strokeStyle = '#d9e0e6';
      ctx.lineWidth = 1;
      ctx.font = '12px Microsoft YaHei, Segoe UI, Arial';
      ctx.fillStyle = '#66727c';
      for (let i = 0; i <= 4; i++) {
        const yy = pad.top + ih * i / 4;
        const val = max - (max - min) * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(w - pad.right, yy); ctx.stroke();
        ctx.fillText(val.toFixed(0), 8, yy + 4);
      }
      const grad = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
      grad.addColorStop(0, 'rgba(17,106,123,.24)');
      grad.addColorStop(1, 'rgba(17,106,123,0)');
      ctx.beginPath();
      values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
      ctx.strokeStyle = '#116a7b';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#66727c';
      const first = historyData.dates[0] || '';
      const last = historyData.dates[historyData.dates.length - 1] || '';
      ctx.fillText(first, pad.left, h - 10);
      ctx.fillText(last, w - pad.right - ctx.measureText(last).width, h - 10);
    }

    function colorScale(t) {
      const stops = [
        [244, 249, 248],
        [154, 204, 197],
        [17, 106, 123],
        [199, 81, 70],
      ];
      const p = Math.max(0, Math.min(1, t)) * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(p));
      const f = p - i;
      const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    function bindSurfaceInteraction(id) {
      const canvas = $(id);
      const view = surfaceViews[id];
      canvas.addEventListener('pointerdown', ev => {
        view.dragging = true;
        view.lastX = ev.clientX;
        view.lastY = ev.clientY;
        canvas.setPointerCapture(ev.pointerId);
      });
      canvas.addEventListener('pointermove', ev => {
        if (view.dragging) {
          const dx = ev.clientX - view.lastX;
          const dy = ev.clientY - view.lastY;
          view.yaw += dx * 0.01;
          view.pitch = Math.max(0.18, Math.min(1.12, view.pitch + dy * 0.008));
          view.lastX = ev.clientX;
          view.lastY = ev.clientY;
          drawSurface(id, id === 'ivSurface' ? surfaceData && surfaceData.iv : surfaceData && surfaceData.local_vol);
          return;
        }
        const rect = canvas.getBoundingClientRect();
        view.mouse = { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
        drawSurface(id, id === 'ivSurface' ? surfaceData && surfaceData.iv : surfaceData && surfaceData.local_vol);
      });
      canvas.addEventListener('pointerup', ev => {
        view.dragging = false;
        try { canvas.releasePointerCapture(ev.pointerId); } catch (_) {}
      });
      canvas.addEventListener('pointerleave', () => {
        view.dragging = false;
        view.mouse = null;
        view.hover = null;
        drawSurface(id, id === 'ivSurface' ? surfaceData && surfaceData.iv : surfaceData && surfaceData.local_vol);
      });
    }

    function drawSurface(id, rows) {
      if (!rows || !surfaceData) return;
      const { ctx, w, h } = setupCanvas(id, 390);
      const view = surfaceViews[id];
      const pad = { left: 42, right: 72, top: 32, bottom: 56 };
      const nr = rows.length;
      const nc = rows[0].length;
      const vals = rows.flat().map(Number);
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      ctx.font = '12px Microsoft YaHei, Segoe UI, Arial';

      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const center = { x: pad.left + plotW * 0.5, y: pad.top + plotH * 0.56 };
      const norm = v => (v - min) / Math.max(max - min, 1e-9);
      const yaw = view.yaw;
      const pitch = view.pitch;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const rotate = p => {
        const x1 = p.x * cy - p.y * sy;
        const y1 = p.x * sy + p.y * cy;
        const z1 = p.z;
        return { x: x1, y: y1 * cp - z1 * sp, z: y1 * sp + z1 * cp };
      };
      const toPoint = (r, c, v) => ({
        x: c / Math.max(nc - 1, 1) - 0.5,
        y: r / Math.max(nr - 1, 1) - 0.5,
        z: norm(v) * 0.58,
        value: v,
        r,
        c,
      });

      const rotated = [];
      for (let r = 0; r < nr; r++) {
        for (let c = 0; c < nc; c++) rotated.push(rotate(toPoint(r, c, rows[r][c])));
      }
      const minX = Math.min(...rotated.map(p => p.x));
      const maxX = Math.max(...rotated.map(p => p.x));
      const minY = Math.min(...rotated.map(p => p.y));
      const maxY = Math.max(...rotated.map(p => p.y));
      const scale = Math.min(plotW / Math.max(maxX - minX, 1e-9), plotH / Math.max(maxY - minY, 1e-9)) * 0.72;
      const project3 = p => {
        const q = rotate(p);
        return { x: center.x + q.x * scale, y: center.y - q.y * scale, depth: q.z, data: p };
      };
      const project = (r, c, v) => project3(toPoint(r, c, v));

      const axis = {
        o: project3({ x: -0.5, y: -0.5, z: 0 }),
        x: project3({ x: 0.62, y: -0.5, z: 0 }),
        y: project3({ x: -0.5, y: 0.62, z: 0 }),
        z: project3({ x: -0.5, y: -0.5, z: 0.72 }),
      };

      ctx.fillStyle = '#fbfcfd';
      ctx.fillRect(pad.left, pad.top, plotW, plotH);
      ctx.strokeStyle = '#edf1f4';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const yy = pad.top + (plotH * i) / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + plotW, yy); ctx.stroke();
      }

      const cells = [];
      for (let r = 0; r < nr - 1; r++) {
        for (let c = 0; c < nc - 1; c++) {
          const avg = (rows[r][c] + rows[r][c + 1] + rows[r + 1][c] + rows[r + 1][c + 1]) / 4;
          const depth = (
            project(r, c, rows[r][c]).depth +
            project(r, c + 1, rows[r][c + 1]).depth +
            project(r + 1, c + 1, rows[r + 1][c + 1]).depth +
            project(r + 1, c, rows[r + 1][c]).depth
          ) / 4;
          cells.push({ r, c, avg, depth });
        }
      }
      cells.sort((a, b) => a.depth - b.depth);
      for (const cell of cells) {
        const r = cell.r;
        const c = cell.c;
        const p1 = project(r, c, rows[r][c]);
        const p2 = project(r, c + 1, rows[r][c + 1]);
        const p3 = project(r + 1, c + 1, rows[r + 1][c + 1]);
        const p4 = project(r + 1, c, rows[r + 1][c]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.lineTo(p4.x, p4.y);
        ctx.closePath();
        ctx.fillStyle = colorScale((cell.avg - min) / Math.max(max - min, 1e-9));
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.82)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      ctx.strokeStyle = 'rgba(23,32,38,.42)';
      ctx.lineWidth = 0.9;
      for (let r = 0; r < nr; r++) {
        ctx.beginPath();
        for (let c = 0; c < nc; c++) {
          const p = project(r, c, rows[r][c]);
          c ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        }
        ctx.stroke();
      }
      for (let c = 0; c < nc; c++) {
        ctx.beginPath();
        for (let r = 0; r < nr; r++) {
          const p = project(r, c, rows[r][c]);
          r ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        }
        ctx.stroke();
      }

      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#172026';
      const drawAxis = (a, b, label, dx, dy) => {
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.fillStyle = '#172026';
        ctx.fillText(label, b.x + dx, b.y + dy);
      };
      drawAxis(axis.o, axis.x, 'K/S0', 8, 4);
      drawAxis(axis.o, axis.y, 'T', 6, -4);
      drawAxis(axis.o, axis.z, 'vol', -18, -8);

      ctx.fillStyle = '#66727c';
      const firstK = surfaceData.strike_ratios[0];
      const midK = surfaceData.strike_ratios[Math.floor(surfaceData.strike_ratios.length / 2)];
      const lastK = surfaceData.strike_ratios[surfaceData.strike_ratios.length - 1];
      const kA = project3({ x: -0.5, y: -0.58, z: 0 });
      const kB = project3({ x: 0, y: -0.58, z: 0 });
      const kC = project3({ x: 0.5, y: -0.58, z: 0 });
      ctx.fillText((firstK * 100).toFixed(0) + '%', kA.x - 14, kA.y + 18);
      ctx.fillText((midK * 100).toFixed(0) + '%', kB.x - 14, kB.y + 18);
      ctx.fillText((lastK * 100).toFixed(0) + '%', kC.x - 14, kC.y + 18);
      const t0 = Number(surfaceData.maturities[0]).toFixed(2) + 'y';
      const t1 = Number(surfaceData.maturities[surfaceData.maturities.length - 1]).toFixed(2) + 'y';
      const tA = project3({ x: -0.58, y: -0.5, z: 0 });
      const tB = project3({ x: -0.58, y: 0.5, z: 0 });
      ctx.fillText(t0, tA.x - 38, tA.y + 6);
      ctx.fillText(t1, tB.x - 38, tB.y + 6);

      const barX = w - pad.right + 18;
      const barY = pad.top + 18;
      const barH = plotH - 36;
      const grad = ctx.createLinearGradient(0, barY + barH, 0, barY);
      for (let i = 0; i <= 20; i++) grad.addColorStop(i / 20, colorScale(i / 20));
      ctx.fillStyle = grad;
      ctx.fillRect(barX, barY, 12, barH);
      ctx.strokeStyle = '#d9e0e6';
      ctx.strokeRect(barX, barY, 12, barH);
      ctx.fillStyle = '#66727c';
      ctx.fillText(fmtPct(max), barX - 8, barY - 6);
      ctx.fillText(fmtPct(min), barX - 8, barY + barH + 16);
      ctx.fillStyle = '#172026';
      ctx.fillText(`min ${fmtPct(min)}   max ${fmtPct(max)}`, pad.left + 4, pad.top + plotH + 24);

      let nearest = null;
      if (view.mouse) {
        for (let r = 0; r < nr; r++) {
          for (let c = 0; c < nc; c++) {
            const p = project(r, c, rows[r][c]);
            const d2 = (p.x - view.mouse.x) ** 2 + (p.y - view.mouse.y) ** 2;
            if (!nearest || d2 < nearest.d2) nearest = { r, c, p, d2, v: rows[r][c] };
          }
        }
      }
      if (nearest && nearest.d2 < 900) {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#172026';
        ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(nearest.p.x, nearest.p.y, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        const k = surfaceData.strike_ratios[nearest.c];
        const t = surfaceData.maturities[nearest.r];
        const lines = [`K/S0 ${(k * 100).toFixed(1)}%`, `T ${Number(t).toFixed(3)}y`, `vol ${fmtPct(nearest.v)}`];
        const boxW = 112;
        const boxH = 62;
        const bx = Math.min(Math.max(nearest.p.x + 10, 8), w - boxW - 8);
        const by = Math.min(Math.max(nearest.p.y - 36, 8), h - boxH - 8);
        ctx.fillStyle = 'rgba(255,255,255,.96)';
        ctx.strokeStyle = '#d9e0e6';
        ctx.fillRect(bx, by, boxW, boxH);
        ctx.strokeRect(bx, by, boxW, boxH);
        ctx.fillStyle = '#172026';
        lines.forEach((line, i) => ctx.fillText(line, bx + 8, by + 18 + i * 16));
      }
    }

    run();
  </script>
</body>
</html>
"""


def _float(qs, key, default):
    return float(qs.get(key, [default])[0])


def _int(qs, key, default):
    return int(float(qs.get(key, [default])[0]))


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/price":
            try:
                self._send_json(self._price(parse_qs(parsed.query)))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, body, content_type, status=200):
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, body, status=200):
        self._send(json.dumps(body, ensure_ascii=False), "application/json; charset=utf-8", status)

    def _price(self, qs):
        index_key = qs.get("index", ["csi500"])[0]
        if index_key not in INDEX_UNIVERSE:
            raise ValueError("unknown index")

        snapshot = load_market_snapshot(index_key)
        r = _float(qs, "r", 0.02)
        q = _float(qs, "q", 0.0)
        iv_source = "历史实现波动率生成的演示 IV 曲面"
        iv_asof = snapshot.asof
        iv_points = "demo"
        if index_key in ETF_OPTION_UNIVERSE:
            try:
                option_surface = load_sse_option_iv_surface(index_key, snapshot.spot)
                surface = option_surface.surface
                iv_source = option_surface.source
                iv_asof = option_surface.asof
                iv_points = f"{option_surface.filtered_points}/{option_surface.raw_points}"
            except Exception as exc:
                surface = build_demo_iv_surface(snapshot.spot, snapshot.annual_vol)
                iv_source = f"真实 IV 加载失败，已退回演示曲面：{exc}"
        else:
            surface = build_demo_iv_surface(snapshot.spot, snapshot.annual_vol)
        lv = LocalVolPricer(surface, snapshot.spot, r, q)
        terms = SnowballTerms(
            notional=_float(qs, "notional", 1_000_000.0),
            maturity_years=_float(qs, "maturity", 1.0),
            coupon=_float(qs, "coupon", 0.16),
            knock_out=_float(qs, "ko", 1.03),
            knock_in=_float(qs, "ki", 0.75),
        )
        paths = min(max(_int(qs, "paths", 5000), 1000), 30000)

        mc = SnowballMCPricer(lv, r, q)
        par_terms = SnowballTerms(
            notional=terms.notional,
            maturity_years=terms.maturity_years,
            coupon=0.0,
            knock_out=terms.knock_out,
            knock_in=terms.knock_in,
            observation_frequency=terms.observation_frequency,
            steps_per_year=terms.steps_per_year,
        )
        par_result = mc.fair_coupon(snapshot.spot, par_terms, paths=paths)
        quoted_result = mc.price(snapshot.spot, terms, paths=paths)
        pde = PDEPricer(lv, r, q, space_steps=120, time_steps=120)
        atm_call = pde.price_european(snapshot.spot, snapshot.spot, min(terms.maturity_years, 1.0), True)

        close = np.asarray(snapshot.close[-260:], dtype=float)
        return {
            "name": snapshot.name,
            "code": snapshot.code,
            "spot": snapshot.spot,
            "annual_vol": snapshot.annual_vol,
            "source": snapshot.source,
            "asof": snapshot.asof,
            "iv_source": iv_source,
            "iv_asof": iv_asof,
            "iv_points": iv_points,
            "fair_coupon": par_result["fair_coupon"],
            "par_price": par_result["price"],
            "quoted_coupon": terms.coupon,
            "quoted_price": quoted_result["price"],
            "std_error": par_result["std_error"],
            "ko_probability": par_result["ko_probability"],
            "ki_probability": par_result["ki_probability"],
            "bonus_probability": par_result["bonus_probability"],
            "no_ki_ko_probability": par_result["no_ki_ko_probability"],
            "ki_no_ko_probability": par_result["ki_no_ko_probability"],
            "ki_ko_probability": par_result["ki_ko_probability"],
            "probability_total": par_result["probability_total"],
            "avg_ko_time": par_result["avg_ko_time"],
            "principal_pv": par_result["principal_pv"],
            "coupon_annuity_pv": par_result["coupon_annuity_pv"],
            "european_call": float(atm_call),
            "atm_local_vol": lv.local_vol(snapshot.spot, min(terms.maturity_years, 1.0)),
            "history": {"dates": snapshot.dates[-260:], "close": close.round(4).tolist()},
            "surface": surface_payload(surface, lv),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    host = args.host
    port = args.port
    while True:
        try:
            server = ThreadingHTTPServer((host, port), DemoHandler)
            break
        except OSError:
            port += 1

    print(f"Demo running at http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
