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


LEGACY_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Vol Snowball Coupon Demo</title>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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
    .math { overflow-x: auto; padding: 10px 12px; margin: 8px 0; border: 1px solid var(--line); border-radius: 6px; background: #fff; font-size: 15px; }
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
      <label>锁定期 月
        <input id="lockout" type="number" value="0" step="1" min="0" max="24">
      </label>
      <label>敲入观察频率
        <select id="kiObs">
          <option value="daily">每日</option>
          <option value="maturity">仅期末 欧式</option>
        </select>
      </label>
      <label>敲出观察频率
        <select id="koObs">
          <option value="monthly">每月</option>
          <option value="daily">每日</option>
          <option value="quarterly">每季度</option>
          <option value="maturity">仅期末</option>
        </select>
      </label>
      <label>是否降敲
        <select id="stepDownEnabled">
          <option value="false">否</option>
          <option value="true">是</option>
        </select>
      </label>
      <label>降敲幅度 每次观察
        <input id="stepDown" type="number" value="0.005" step="0.001" min="0" max="0.1">
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
          <p><strong>1. 从市场期权价格得到隐含波动率曲面</strong></p>
          <p>市场上每个到期日 \(T\) 和执行价 \(K\) 的 vanilla option 都有价格。把这些价格代入 Black-Scholes 反解，就得到隐含波动率曲面 \(\sigma_{\mathrm{imp}}(K,T)\)。它不是历史波动率，而是市场今天对未来分布的定价。</p>
          <div class="math">\[
            C^{\mathrm{mkt}}(K,T)
            =
            BSCall\!\left(S_0,K,T,r,q,\sigma_{\mathrm{imp}}(K,T)\right)
          \]</div>
          <p>ETF 模式下，本 demo 使用上交所风险指标中的 <code>IMPLC_VOLATLTY</code> 作为 \(\sigma_{\mathrm{imp}}\) 的市场输入，并把离散合约点插值成规则曲面。</p>

          <p><strong>2. Local volatility 假设</strong></p>
          <p>Black-Scholes 使用一个常数波动率，而 local volatility 允许瞬时波动率随价格和时间变化。在风险中性测度 \(\mathbb Q\) 下，标的过程写成：</p>
          <div class="math">\[
            \frac{dS_t}{S_t}
            = (r-q)\,dt + \sigma_{\mathrm{loc}}(S_t,t)\,dW_t^{\mathbb Q}
          \]</div>
          <p>这里 \(\sigma_{\mathrm{loc}}(S,t)\) 的目标不是预测真实波动率，而是在风险中性定价框架下复现市场 vanilla option 曲面。</p>

          <p><strong>3. Dupire 公式：从价格曲面到局部波动率曲面</strong></p>
          <p>把隐含波动率曲面转成欧式看涨期权价格曲面 \(C(K,T)\) 后，可以用 Dupire 公式得到局部波动率：</p>
          <div class="math">\[
            \sigma_{\mathrm{loc}}^2(K,T)
            =
            \frac{
              \frac{\partial C}{\partial T}
              + qC
              + (r-q)K\frac{\partial C}{\partial K}
            }{
              \frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}
            }
          \]</div>
          <p>直觉上，\(\partial C/\partial T\) 描述期限方向的价格变化，\(\partial C/\partial K\) 和 \(\partial^2 C/\partial K^2\) 描述执行价方向的曲率和隐含分布。实际代码里会先平滑曲面，再用有限差分近似这些偏导。</p>

          <p><strong>4. 用 local volatility 生成风险中性路径</strong></p>
          <p>拿到 \(\sigma_{\mathrm{loc}}\) 后，用 Euler/log-Euler 方式离散模拟路径：</p>
          <div class="math">\[
            S_{t+\Delta t}
            =
            S_t\exp\!\left(
              (r-q-\tfrac{1}{2}\sigma_{\mathrm{loc}}^2)\Delta t
              + \sigma_{\mathrm{loc}}\sqrt{\Delta t}\,Z
            \right),
            \quad Z\sim N(0,1)
          \]</div>
          <p>每条路径都会按照用户选择的雪球条款检查敲入、敲出、锁定期和降敲。观察频率越高，触发机会越多；锁定期会推迟可敲出时间；降敲会让后续敲出线逐步降低。</p>

          <p><strong>5. 四类互斥路径</strong></p>
          <p>为了理解风险，最终路径分成四类，并且概率总和为 1：</p>
          <div class="math">\[
            P_{\mathrm{bonus}}
            + P_{\mathrm{noKI,KO}}
            + P_{\mathrm{KI,noKO}}
            + P_{\mathrm{KI,KO}}
            = 1
          \]</div>
          <p>\(P_{\mathrm{bonus}}\) 是未敲入且未敲出，也就是持有到期获得票息且没有本金亏损的路径占比。</p>

          <p><strong>6. 公允票息的数学定义</strong></p>
          <p>令年化票息为 \(c\)，每条路径对应一个贴现后的总现金流 \(X_i(c)\)。公允票息定义为使产品理论现值等于名义本金的 \(c\)：</p>
          <div class="math">\[
            PV(c)
            =
            \mathbb E^{\mathbb Q}\!\left[X(c)\right]
            =
            \mathrm{Notional}
          \]</div>
          <p>在 Monte Carlo 中，用 \(N\) 条路径近似这个期望：</p>
          <div class="math">\[
            PV(c)
            \approx
            \frac{1}{N}\sum_{i=1}^{N}X_i(c)
          \]</div>
          <p>由于雪球票息现金流对 \(c\) 是线性的，可以把路径现金流写成：</p>
          <div class="math">\[
            X_i(c)=A_i+cB_i
          \]</div>
          <p>其中 \(A_i\) 是路径中与票息无关的贴现现金流，\(B_i\) 是该路径对单位年化票息的贴现敏感度。因此：</p>
          <div class="math">\[
            c_{\mathrm{fair}}
            =
            \frac{
              \mathrm{Notional}
              -
              \frac{1}{N}\sum_{i=1}^{N}A_i
            }{
              \frac{1}{N}\sum_{i=1}^{N}B_i
            }
          \]</div>
          <p>这就是页面中“公允年化票息”的来源：不是手动调参，也不是历史收益率，而是在 local volatility 风险中性路径下让理论价值等于名义本金的解。</p>
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
    function fmtAxis(x) {
      const v = Math.abs(Number(x));
      if (v < 10) return Number(x).toFixed(3);
      if (v < 100) return Number(x).toFixed(2);
      return Number(x).toFixed(0);
    }
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
        lockout: $('lockout').value,
        ki_obs: $('kiObs').value,
        ko_obs: $('koObs').value,
        step_down_enabled: $('stepDownEnabled').value,
        step_down: $('stepDown').value,
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
          ['历史年化波动率', data.uses_real_iv ? '真实IV模式未用于定价' : fmtPct(data.annual_vol)],
          ['手动报价票息', fmtPct(data.quoted_coupon)],
          ['锁定期', data.terms.lockout_months + ' 月'],
          ['敲入观察', data.terms.knock_in_observation],
          ['敲出观察', data.terms.knock_out_observation],
          ['降敲', data.terms.step_down_enabled ? ('是，每次 ' + fmtPct(data.terms.step_down)) : '否'],
          ['公允票息下PV', fmtMoney(data.par_price)],
          ['ATM局部波动率', fmtPct(data.atm_local_vol)],
          ['红利概率：未敲入未敲出', fmtPct(data.bonus_probability)],
          ['未敲入后敲出', fmtPct(data.no_ki_ko_probability)],
          ['敲入未敲出', fmtPct(data.ki_no_ko_probability)],
          ['敲入后敲出', fmtPct(data.ki_ko_probability)],
          ['四类概率合计', fmtPct(data.probability_total)],
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
      if (panel.classList.contains('open') && window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([panel]);
      }
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
        ctx.fillText(fmtAxis(val), 8, yy + 4);
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

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>雪球本地波动率定价工作台</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172026;
      --muted: #66727c;
      --line: #d9e0e6;
      --fill: #f6f8fa;
      --brand: #0f6f7f;
      --brand-soft: #e8f4f5;
      --good: #1f7a4f;
      --warn: #b7791f;
      --bad: #b94a48;
      --surface: #fff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: #f2f5f7;
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 15px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 13px; color: var(--muted); font-weight: 700; }
    .sub { margin-top: 6px; color: var(--muted); font-size: 13px; }
    .status { color: var(--muted); font-size: 13px; min-height: 20px; text-align: right; }
    main {
      padding: 16px 24px 24px;
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 16px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .controls {
      padding: 14px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .group {
      display: grid;
      gap: 10px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line);
    }
    .group:last-of-type { border-bottom: 0; padding-bottom: 0; }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select {
      width: 100%;
      min-height: 36px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    .inline {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--brand);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: .65; cursor: wait; }
    .workspace { display: grid; gap: 14px; min-width: 0; }
    .summary {
      display: grid;
      grid-template-columns: minmax(280px, 1.2fr) minmax(240px, .8fr);
      gap: 14px;
    }
    .hero {
      padding: 18px;
      background: var(--brand-soft);
      border-color: #b9d9dd;
    }
    .hero-line {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .hero-value {
      font-size: 34px;
      color: var(--brand);
      font-weight: 800;
      white-space: nowrap;
    }
    .verdict {
      margin-top: 10px;
      padding: 9px 10px;
      border-radius: 6px;
      background: #fff;
      border: 1px solid #cfe3e6;
      font-size: 13px;
      line-height: 1.5;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 78px;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 19px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .table-panel { padding: 14px; }
    .details {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .details td {
      padding: 8px 6px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }
    .details td:first-child { color: var(--muted); width: 45%; }
    .details td:last-child { text-align: right; }
    .risk-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .risk-card {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #edf1f4;
      overflow: hidden;
      margin-top: 8px;
    }
    .bar > i { display: block; height: 100%; background: var(--brand); }
    .diagnostic {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .diagnostic.good { border-color: #b8dec9; background: #f4fbf7; }
    .diagnostic.warn { border-color: #ead49a; background: #fffaf0; }
    .diagnostic.bad { border-color: #e6b8b8; background: #fff6f6; }
    .diag-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .tabs {
      display: flex;
      gap: 6px;
      padding: 10px 10px 0;
      border-bottom: 1px solid var(--line);
      background: #fff;
      border-radius: 8px 8px 0 0;
    }
    .tab {
      padding: 8px 10px;
      border-radius: 6px 6px 0 0;
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
    }
    .tab.active {
      background: var(--brand-soft);
      color: var(--brand);
      font-weight: 700;
    }
    .tab-content { display: none; padding: 12px; }
    .tab-content.active { display: block; }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .chart-wrap, .surface-wrap {
      min-height: 380px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    canvas { width: 100%; max-height: 360px; }
    .surface-wrap canvas { cursor: grab; max-height: 420px; }
    .surface-wrap canvas:active { cursor: grabbing; }
    .note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      margin-top: 10px;
    }
    .lecture {
      max-width: 980px;
      color: #26323a;
      font-size: 14px;
      line-height: 1.75;
    }
    .lecture h2 { margin-top: 0; font-size: 18px; }
    .lecture h3 {
      margin: 18px 0 8px;
      color: var(--ink);
      font-size: 15px;
    }
    .lecture p { margin: 8px 0; }
    .lecture ul, .lecture ol { margin: 8px 0 8px 22px; padding: 0; }
    .formula {
      overflow-x: auto;
      margin: 10px 0;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      font-family: Cambria Math, STIX Two Math, Times New Roman, serif;
      font-size: 16px;
      line-height: 1.7;
      white-space: nowrap;
    }
    .formula .frac {
      display: inline-grid;
      grid-template-rows: auto auto;
      vertical-align: middle;
      text-align: center;
      line-height: 1.25;
      margin: 0 3px;
    }
    .formula .frac > span:first-child {
      border-bottom: 1px solid currentColor;
      padding: 0 4px 2px;
    }
    .formula .frac > span:last-child { padding: 2px 4px 0; }
    .formula small { font-size: 85%; }
    .term-table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 13px;
    }
    .term-table th, .term-table td {
      border-top: 1px solid var(--line);
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
    }
    .term-table th { color: var(--muted); font-weight: 700; }
    .callout {
      margin: 12px 0;
      padding: 10px 12px;
      border-left: 4px solid var(--brand);
      background: var(--brand-soft);
      border-radius: 4px;
    }
    .empty {
      padding: 22px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fff;
    }
    @media (max-width: 1180px) {
      .summary, .chart-grid { grid-template-columns: 1fr; }
      .metric-grid, .risk-grid, .diag-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 900px) {
      header { display: block; padding: 16px 14px 10px; }
      .status { text-align: left; margin-top: 8px; }
      main { grid-template-columns: 1fr; padding: 12px; }
      .inline { grid-template-columns: 1fr; }
      .metric-grid, .risk-grid, .diag-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>雪球本地波动率定价工作台</h1>
      <div class="sub">用市场 IV 或演示曲面校准 Dupire 局部波动率，反解雪雪球公允年化票息并检查模型稳定性。</div>
    </div>
    <div class="status" id="status">准备就绪</div>
  </header>

  <main>
    <section class="panel controls">
      <div class="group">
        <h3>标的与模型</h3>
        <label>标的
          <select id="index">
            <optgroup label="ETF 真实 IV">
              <option value="etf50">50ETF</option>
              <option value="etf300">300ETF</option>
              <option value="etf500">500ETF</option>
              <option value="kc50">科创50ETF</option>
            </optgroup>
            <optgroup label="指数演示曲面">
              <option value="csi500" selected>中证500</option>
              <option value="csi1000">中证1000</option>
              <option value="csi300">沪深300</option>
              <option value="sse50">上证50</option>
            </optgroup>
          </select>
        </label>
        <div class="inline">
          <label>无风险利率 r
            <input id="r" type="number" value="0.02" step="0.005" min="-0.05" max="0.2">
          </label>
          <label>分红率 q
            <input id="q" type="number" value="0" step="0.005" min="-0.05" max="0.2">
          </label>
        </div>
      </div>

      <div class="group">
        <h3>产品条款</h3>
        <label>名义本金
          <input id="notional" type="number" value="1000000" step="10000" min="10000">
        </label>
        <div class="inline">
          <label>期限 年
            <input id="maturity" type="number" value="1" step="0.25" min="0.25" max="2">
          </label>
          <label>报价票息 年化
            <input id="coupon" type="number" value="0.16" step="0.01" min="-0.5" max="1">
          </label>
        </div>
        <div class="inline">
          <label>敲出线 S0 倍数
            <input id="ko" type="number" value="1.03" step="0.01" min="0.5" max="2">
          </label>
          <label>敲入线 S0 倍数
            <input id="ki" type="number" value="0.75" step="0.01" min="0.1" max="1.5">
          </label>
        </div>
      </div>

      <div class="group">
        <h3>观察规则</h3>
        <div class="inline">
          <label>锁定期 月
            <input id="lockout" type="number" value="0" step="1" min="0" max="24">
          </label>
          <label>模拟路径
            <input id="paths" type="number" value="3000" step="1000" min="1000" max="30000">
          </label>
        </div>
        <div class="inline">
          <label>敲入观察
            <select id="kiObs">
              <option value="daily">每日</option>
              <option value="maturity">仅到期</option>
            </select>
          </label>
          <label>敲出观察
            <select id="koObs">
              <option value="monthly">每月</option>
              <option value="daily">每日</option>
              <option value="quarterly">每季度</option>
              <option value="maturity">仅到期</option>
            </select>
          </label>
        </div>
        <div class="inline">
          <label>是否降敲
            <select id="stepDownEnabled">
              <option value="false">否</option>
              <option value="true">是</option>
            </select>
          </label>
          <label>每次降敲幅度
            <input id="stepDown" type="number" value="0.005" step="0.001" min="0" max="0.1">
          </label>
        </div>
      </div>

      <button id="run">重新定价</button>
    </section>

    <section class="workspace">
      <section class="summary">
        <div class="panel hero">
          <div class="hero-line">
            <div>
              <h2>模型公允年化票息</h2>
              <div class="sub" id="quoteMeta">等待计算</div>
            </div>
            <div class="hero-value" id="fairCoupon">-</div>
          </div>
          <div class="verdict" id="verdict">输入条款后，系统会比较报价票息与模型公允票息，并给出 PV 差异。</div>
        </div>
        <div class="panel table-panel">
          <h2>定价摘要</h2>
          <table class="details"><tbody id="summaryRows"></tbody></table>
        </div>
      </section>

      <section class="metric-grid" id="metrics"></section>

      <section class="risk-grid" id="riskCards"></section>

      <section class="diagnostic" id="diagnosticBox">
        <h2>模型诊断</h2>
        <div id="diagnosticText">等待计算局部波动率曲面诊断。</div>
        <div class="diag-grid" id="diagnosticMetrics"></div>
      </section>

      <section class="panel">
        <div class="tabs">
          <div class="tab active" data-tab="overview">行情</div>
          <div class="tab" data-tab="surfaces">曲面</div>
          <div class="tab" data-tab="details">明细</div>
          <div class="tab" data-tab="model">模型说明</div>
        </div>
        <div class="tab-content active" id="tab-overview">
          <section class="chart-wrap">
            <h2>收盘价走势</h2>
            <canvas id="chart"></canvas>
          </section>
        </div>
        <div class="tab-content" id="tab-surfaces">
          <div class="chart-grid">
            <section class="surface-wrap">
              <h2>输入隐含波动率曲面 IV(K,T)</h2>
              <canvas id="ivSurface"></canvas>
            </section>
            <section class="surface-wrap">
              <h2>Dupire 局部波动率曲面 σloc(S,t)</h2>
              <canvas id="lvSurface"></canvas>
            </section>
          </div>
        </div>
        <div class="tab-content" id="tab-details">
          <div class="table-panel">
            <h2>数据与条款明细</h2>
            <table class="details"><tbody id="details"></tbody></table>
            <div class="note">ETF 期权 IV 来自上交所期权风险指标 IMPLC_VOLATLTY；若实时接口不可用，会使用本地缓存或演示曲面。指数标的当前仍使用历史波动率生成的演示 IV 曲面。</div>
          </div>
        </div>
        <div class="tab-content" id="tab-model">
          <div class="table-panel lecture">
            <h2>模型说明讲义：从 IV 曲面到雪球公允票息</h2>
            <p>这一页的定价逻辑可以分成四步：先用市场或演示隐含波动率曲面构造欧式期权价格曲面，再用 Dupire 公式得到局部波动率曲面，然后在风险中性测度下模拟雪球路径，最后用平价条件反解年化票息。下面按这个顺序说明。</p>

            <h3>1. 风险中性定价的出发点</h3>
            <p>权益类标的在风险中性测度下，漂移不是历史平均收益率，而是无风险利率扣除连续分红率。代码采用的局部波动率模型是：</p>
            <div class="formula">
              dS<sub>t</sub> / S<sub>t</sub> = (r - q) dt + σ<sub>loc</sub>(S<sub>t</sub>, t) dW<sub>t</sub><sup>Q</sup>
            </div>
            <table class="term-table">
              <thead><tr><th>符号</th><th>含义</th><th>在代码中的位置</th></tr></thead>
              <tbody>
                <tr><td>S<sub>t</sub></td><td>标的价格过程</td><td>Monte Carlo 路径中的当前价格</td></tr>
                <tr><td>r</td><td>连续复利无风险利率</td><td>页面输入框“无风险利率 r”</td></tr>
                <tr><td>q</td><td>连续分红率或等价收益率</td><td>页面输入框“分红率 q”</td></tr>
                <tr><td>σ<sub>loc</sub>(S,t)</td><td>局部波动率，随价格和时间变化</td><td><code>LocalVolPricer.local_vol</code></td></tr>
              </tbody>
            </table>
            <p>风险中性定价的核心是：贴现后的资产价格和衍生品价格应由风险中性期望给出。因此任意现金流 X 的理论现值写成：</p>
            <div class="formula">
              PV = E<sup>Q</sup>[ exp(-r τ) X ]
            </div>
            <p>这里的 Q 表示风险中性测度。注意，这不是用历史收益率预测未来，而是在无套利框架下用市场期权价格隐含出的分布来定价。</p>

            <h3>2. 市场价格、隐含波动率与 Dupire 输入</h3>
            <p>严格地说，隐含波动率不是原始定义上的市场价格。原始对象是每个执行价 K、到期 T 的 vanilla option 市场价格 C<sub>mkt</sub>(K,T)。所谓隐含波动率，是把这个市场价格代入带连续分红率的 Black-Scholes 公式后，反解出来的那个常数波动率：</p>
            <div class="formula">
              C<sub>mkt</sub>(K,T) = BS(S<sub>0</sub>, K, T, r, q, σ<sub>imp</sub>)
            </div>
            <p>因此，从概念定义上讲，方向是：</p>
            <div class="formula">
              C<sub>mkt</sub>(K,T) → 反解 Black-Scholes → σ<sub>imp</sub>(K,T)
            </div>
            <p>本项目页面和交易所期权风险指标里经常直接拿到的是 IV 字段，所以代码把已经得到的隐含波动率曲面记为：</p>
            <div class="formula">
              σ<sub>imp</sub> = σ<sub>imp</sub>(K, T)
            </div>
            <p>但 Dupire 公式本身需要的是欧式看涨期权价格曲面及其偏导数，而不是 IV 对 K、T 的偏导数。于是工程实现会把 IV 重新代入同一个 Black-Scholes 映射，恢复出与该 IV 等价的看涨期权价格曲面 C(K,T)：</p>
            <div class="formula">
              C(K,T) = S<sub>0</sub> exp(-qT) N(d<sub>1</sub>) - K exp(-rT) N(d<sub>2</sub>)
            </div>
            <div class="formula">
              d<sub>1</sub> =
              <span class="frac"><span>ln(S<sub>0</sub>/K) + (r - q + 0.5 σ<sub>imp</sub><sup>2</sup>)T</span><span>σ<sub>imp</sub> √T</span></span>,
              &nbsp; d<sub>2</sub> = d<sub>1</sub> - σ<sub>imp</sub> √T
            </div>
            <p>这一步不是在定义隐含波动率，而是在把“以 IV 形式保存的市场信息”转回 Dupire 所需的价格坐标。更准确的链条是：</p>
            <div class="formula">
              C<sub>mkt</sub>(K,T) ↔ σ<sub>imp</sub>(K,T) → C(K,T) → ∂<sub>T</sub>C, ∂<sub>K</sub>C, ∂<sub>KK</sub>C → σ<sub>loc</sub>(K,T)
            </div>
            <p>其中第一个双向箭头表示：在给定 S<sub>0</sub>、K、T、r、q 后，只要期权价格没有违反基本无套利边界，价格和 Black-Scholes 隐含波动率可以互相转换。真正进入 Dupire 公式的是价格曲面 C(K,T)，IV 只是市场价格曲面的另一种报价方式。</p>
            <div class="callout">如果 IV 曲面插值不稳定或不满足无套利约束，C(K,T) 关于 K 的凸性可能被破坏，后面的 Dupire 局部波动率就会出现负方差或 fallback。</div>

            <h3>3. Dupire 公式：为什么能从 C(K,T) 得到局部波动率</h3>
            <p>Dupire 的思想是：如果所有欧式看涨期权价格 C(K,T) 都已知，那么这些价格等价地刻画了风险中性边际分布。局部波动率模型希望找到一个 σ<sub>loc</sub>(S,t)，使该模型能复现这些 vanilla option 价格。</p>
            <p>代码使用的 Dupire 公式为：</p>
            <div class="formula">
              σ<sub>loc</sub><sup>2</sup>(K,T) =
              <span class="frac">
                <span>∂<sub>T</sub>C + qC + (r - q)K ∂<sub>K</sub>C</span>
                <span>0.5 K<sup>2</sup> ∂<sub>KK</sub>C</span>
              </span>
            </div>
            <p>各项可以这样理解：</p>
            <ul>
              <li><strong>∂<sub>T</sub>C</strong>：期限方向的价格变化，反映不同到期日之间的时间价值结构。</li>
              <li><strong>∂<sub>K</sub>C</strong>：执行价方向的一阶变化，与尾部分布和远期漂移项有关。</li>
              <li><strong>∂<sub>KK</sub>C</strong>：执行价方向的二阶导数，本质上对应风险中性密度；它必须为正，才有合理的概率解释。</li>
              <li><strong>qC 与 (r-q)K∂<sub>K</sub>C</strong>：连续分红率和风险中性漂移带来的修正项。</li>
            </ul>
            <p>在代码中，这些偏导数用有限差分近似：</p>
            <div class="formula">
              ∂<sub>T</sub>C ≈
              <span class="frac"><span>C(K,T+ΔT) - C(K,T-ΔT)</span><span>2ΔT</span></span>
            </div>
            <div class="formula">
              ∂<sub>K</sub>C ≈
              <span class="frac"><span>C(K+ΔK,T) - C(K-ΔK,T)</span><span>2ΔK</span></span>,
              &nbsp;
              ∂<sub>KK</sub>C ≈
              <span class="frac"><span>C(K+ΔK,T) - 2C(K,T) + C(K-ΔK,T)</span><span>ΔK<sup>2</sup></span>
            </div>
            <p>因此，local vol 诊断中的“正密度比例”就是在检查分母 0.5K<sup>2</sup>∂<sub>KK</sub>C 是否大多为正；“fallback 比例”表示有多少网格点无法稳定使用 Dupire 公式，只能回退到原始 IV。</p>

            <h3>4. 为什么雪球需要路径模拟</h3>
            <p>欧式期权只关心到期时的 S<sub>T</sub>，但雪球关心路径中是否触碰敲入线、敲出线，以及在哪个观察日敲出。因此它不能只用终值分布定价，需要模拟整条路径。</p>
            <p>代码用 log-Euler 方式离散风险中性 SDE：</p>
            <div class="formula">
              S<sub>t+Δt</sub> =
              S<sub>t</sub> exp((r - q - 0.5σ<sub>loc</sub><sup>2</sup>)Δt + σ<sub>loc</sub>√Δt Z),
              &nbsp; Z ~ N(0,1)
            </div>
            <p>每个时间步都会根据当前价格和时间读取 σ<sub>loc</sub>(S,t)，再生成下一步价格。为了让网页响应更快，代码会先在价格网格和较粗的时间网格上预计算 local vol，再对路径价格插值。</p>

            <h3>5. 雪球现金流的路径分类</h3>
            <p>每条路径最终会落入四类之一：</p>
            <table class="term-table">
              <thead><tr><th>路径类型</th><th>含义</th><th>典型现金流</th></tr></thead>
              <tbody>
                <tr><td>未敲入未敲出</td><td>持有到期，未触碰敲入线，也未触发敲出</td><td>返还本金，并按到期累计票息</td></tr>
                <tr><td>未敲入后敲出</td><td>先没有敲入，在某个观察日达到敲出线</td><td>提前返还本金，并按持有时间支付票息</td></tr>
                <tr><td>敲入未敲出</td><td>路径中触碰敲入线，之后没有敲出</td><td>到期按标的跌幅承担本金损失</td></tr>
                <tr><td>敲入后敲出</td><td>先敲入，后续观察日又达到敲出线</td><td>提前返还本金，并按代码中的规则支付累计票息</td></tr>
              </tbody>
            </table>
            <div class="formula">
              P<sub>未敲入未敲出</sub> + P<sub>未敲入后敲出</sub> + P<sub>敲入未敲出</sub> + P<sub>敲入后敲出</sub> = 1
            </div>
            <p>页面中的敲入概率、敲出概率和四类路径概率，就是对这些路径事件做 Monte Carlo 频率估计。</p>

            <h3>6. 公允票息为什么可以直接反解</h3>
            <p>雪球现金流中，本金损益部分与票息 c 无关；票息部分对 c 是线性的。因此每条路径的贴现现金流可以写成：</p>
            <div class="formula">
              X<sub>i</sub>(c) = A<sub>i</sub> + c B<sub>i</sub>
            </div>
            <p>其中 A<sub>i</sub> 是第 i 条路径的贴现本金现金流，B<sub>i</sub> 是单位年化票息对应的贴现累计计息本金。Monte Carlo 估计的产品现值为：</p>
            <div class="formula">
              PV(c) ≈
              <span class="frac"><span>1</span><span>N</span></span>
              Σ<sub>i=1</sub><sup>N</sup> (A<sub>i</sub> + c B<sub>i</sub>)
              = Ā + c B̄
            </div>
            <p>公允票息定义为让理论现值等于名义本金 N<sub>0</sub> 的 c：</p>
            <div class="formula">
              N<sub>0</sub> = Ā + c<sub>fair</sub>B̄
              &nbsp; ⇒ &nbsp;
              c<sub>fair</sub> =
              <span class="frac"><span>N<sub>0</sub> - Ā</span><span>B̄</span></span>
            </div>
            <p>这也是为什么网页可以同时显示“报价票息 PV”和“公允票息”：同一批路径下，换一个票息 c 只是在 Ā + cB̄ 里换斜率项，不需要重新定义产品结构。</p>

            <h3>7. 如何阅读模型诊断</h3>
            <ul>
              <li><strong>fallback 比例低</strong>：说明 Dupire 公式在多数网格点上可用，局部波动率曲面更可信。</li>
              <li><strong>正密度比例低</strong>：说明 C(K,T) 关于 K 的凸性可能被插值破坏，风险中性密度解释变弱。</li>
              <li><strong>local vol 最大值过高或过低</strong>：可能是 IV 曲面边界、插值、期限结构或有限差分导致的数值异常。</li>
            </ul>
            <div class="callout">这套工具适合做本地波动率雪雪球定价原型和模型解释。若用于真实报价，还需要更完整的交易日历、分红/forward 一致性、无套利 IV 曲面校准和逐合同现金流核对。</div>
          </div>
        </div>
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
    function fmtSignedMoney(x) {
      const n = Number(x);
      const sign = n > 0 ? '+' : '';
      return sign + n.toLocaleString('zh-CN', { maximumFractionDigits: 0 });
    }
    function fmtPct(x) { return (Number(x) * 100).toFixed(2) + '%'; }
    function fmtNum(x, digits=3) { return Number(x).toFixed(digits); }
    function fmtAxis(x) {
      const v = Math.abs(Number(x));
      if (v < 10) return Number(x).toFixed(3);
      if (v < 100) return Number(x).toFixed(2);
      return Number(x).toFixed(0);
    }
    function metric(label, value) {
      return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
    }
    function row(label, value) {
      return `<tr><td>${label}</td><td>${value}</td></tr>`;
    }
    function riskCard(label, value, pct) {
      const width = Math.max(0, Math.min(100, Number(pct) * 100));
      return `<div class="risk-card"><span>${label}</span><strong>${value}</strong><div class="bar"><i style="width:${width}%"></i></div></div>`;
    }
    function classifyDiagnostics(d) {
      if (!d) return { cls: '', text: '等待计算局部波动率曲面诊断。' };
      if (d.fallback_ratio <= 0.02) return { cls: 'good', text: '曲面诊断稳定：Dupire 有效点比例较高。' };
      if (d.fallback_ratio <= 0.12) return { cls: 'warn', text: '曲面存在一定数值不稳定：部分点回退到 IV。' };
      return { cls: 'bad', text: '曲面诊断偏弱：fallback 比例较高，local vol 解释需谨慎。' };
    }

    async function run() {
      $('run').disabled = true;
      $('status').textContent = '定价中...';
      const q = new URLSearchParams({
        index: $('index').value,
        r: $('r').value,
        q: $('q').value,
        notional: $('notional').value,
        maturity: $('maturity').value,
        coupon: $('coupon').value,
        ko: $('ko').value,
        ki: $('ki').value,
        lockout: $('lockout').value,
        ki_obs: $('kiObs').value,
        ko_obs: $('koObs').value,
        step_down_enabled: $('stepDownEnabled').value,
        step_down: $('stepDown').value,
        paths: $('paths').value,
      });
      try {
        const res = await fetch('/api/price?' + q.toString());
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || '请求失败');
        render(data);
        $('status').textContent = '完成';
      } catch (err) {
        $('status').textContent = err.message;
      } finally {
        $('run').disabled = false;
      }
    }

    function render(data) {
      const pvDiff = Number(data.quoted_price) - Number(data.terms.notional);
      const couponDiff = Number(data.quoted_coupon) - Number(data.fair_coupon);
      const relDiff = pvDiff / Math.max(Number(data.terms.notional), 1);
      let verdict = '报价票息接近模型公允票息。';
      if (couponDiff > 0.002) verdict = `报价票息高于模型公允 ${fmtPct(couponDiff)}，该票息下 PV 相对本金为 ${fmtSignedMoney(pvDiff)}。`;
      if (couponDiff < -0.002) verdict = `报价票息低于模型公允 ${fmtPct(-couponDiff)}，该票息下 PV 相对本金为 ${fmtSignedMoney(pvDiff)}。`;

      $('fairCoupon').textContent = fmtPct(data.fair_coupon);
      $('quoteMeta').textContent = `${data.name} (${data.code}) · ${data.iv_asof} · ${data.uses_real_iv ? '真实 IV' : '演示 IV'}`;
      $('verdict').textContent = verdict;
      $('summaryRows').innerHTML = [
        row('输入票息', fmtPct(data.quoted_coupon)),
        row('输入票息 PV', fmtMoney(data.quoted_price)),
        row('PV - 本金', `${fmtSignedMoney(pvDiff)} (${fmtPct(relDiff)})`),
        row('Monte Carlo 标准误', fmtMoney(data.std_error)),
      ].join('');

      $('metrics').innerHTML = [
        metric('标的现货', Number(data.spot).toFixed(2)),
        metric('公允票息下 PV', fmtMoney(data.par_price)),
        metric('ATM local vol', fmtPct(data.atm_local_vol)),
        metric('平均敲出时间', data.avg_ko_time ? `${fmtNum(data.avg_ko_time, 3)} 年` : '-'),
      ].join('');

      $('riskCards').innerHTML = [
        riskCard('敲出概率', fmtPct(data.ko_probability), data.ko_probability),
        riskCard('敲入概率', fmtPct(data.ki_probability), data.ki_probability),
        riskCard('敲入未敲出', fmtPct(data.ki_no_ko_probability), data.ki_no_ko_probability),
        riskCard('未敲入未敲出', fmtPct(data.bonus_probability), data.bonus_probability),
      ].join('');

      const d = data.local_vol_diagnostics;
      const diag = classifyDiagnostics(d);
      $('diagnosticBox').className = `diagnostic ${diag.cls}`;
      $('diagnosticText').textContent = diag.text;
      $('diagnosticMetrics').innerHTML = [
        metric('fallback 比例', fmtPct(d.fallback_ratio)),
        metric('正密度比例', fmtPct(d.positive_density_ratio)),
        metric('正分子比例', fmtPct(d.positive_numerator_ratio)),
        metric('最小 local vol', fmtPct(d.min_local_vol)),
        metric('最大 local vol', fmtPct(d.max_local_vol)),
      ].join('');

      $('details').innerHTML = [
        row('标的', `${data.name} (${data.code})`),
        row('行情日期', data.asof),
        row('行情来源', data.source),
        row('IV 来源', data.iv_source),
        row('IV 日期', data.iv_asof),
        row('IV 点数', data.iv_points),
        row('历史年化波动率', data.uses_real_iv ? '真实 IV 模式未用于定价' : fmtPct(data.annual_vol)),
        row('名义本金', fmtMoney(data.terms.notional)),
        row('期限', `${data.terms.maturity_years} 年`),
        row('敲入/敲出', `${fmtPct(data.terms.knock_in)} / ${fmtPct(data.terms.knock_out)}`),
        row('观察规则', `敲入 ${data.terms.knock_in_observation}，敲出 ${data.terms.knock_out_observation}`),
        row('锁定期', `${data.terms.lockout_months} 月`),
        row('降敲', data.terms.step_down_enabled ? `是，每次 ${fmtPct(data.terms.step_down)}` : '否'),
        row('未裁剪公允票息', fmtPct(data.unclipped_fair_coupon)),
        row('票息是否被裁剪', data.fair_coupon_clipped ? '是' : '否'),
      ].join('');

      historyData = data.history;
      surfaceData = data.surface;
      drawAll();
    }

    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
        tab.classList.add('active');
        $(`tab-${tab.dataset.tab}`).classList.add('active');
        drawAll();
      });
    });

    $('run').addEventListener('click', run);
    window.addEventListener('resize', drawAll);
    for (const id of Object.keys(surfaceViews)) bindSurfaceInteraction(id);

    function drawAll() {
      drawChart();
      drawSurface('ivSurface', surfaceData && surfaceData.iv);
      drawSurface('lvSurface', surfaceData && surfaceData.local_vol);
    }

    function setupCanvas(id, height) {
      const canvas = $(id);
      if (!canvas || !canvas.offsetParent) return null;
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
      const setup = setupCanvas('chart', 330);
      if (!setup) return;
      const { ctx, w, h } = setup;
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
        ctx.fillText(fmtAxis(val), 8, yy + 4);
      }
      const grad = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
      grad.addColorStop(0, 'rgba(15,111,127,.22)');
      grad.addColorStop(1, 'rgba(15,111,127,0)');
      ctx.beginPath();
      values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
      ctx.strokeStyle = '#0f6f7f';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#66727c';
      const first = historyData.dates[0] || '';
      const last = historyData.dates[historyData.dates.length - 1] || '';
      ctx.fillText(first, pad.left, h - 10);
      ctx.fillText(last, w - pad.right - ctx.measureText(last).width, h - 10);
    }

    function colorScale(t) {
      const stops = [[246,248,250], [159,210,199], [15,111,127], [185,74,72]];
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
      const setup = setupCanvas(id, 390);
      if (!setup) return;
      const { ctx, w, h } = setup;
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
        return { x: x1, y: y1 * cp - p.z * sp, z: y1 * sp + p.z * cp };
      };
      const toPoint = (r, c, v) => ({
        x: c / Math.max(nc - 1, 1) - 0.5,
        y: r / Math.max(nr - 1, 1) - 0.5,
        z: norm(v) * 0.58,
        value: v, r, c,
      });
      const rotated = [];
      for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) rotated.push(rotate(toPoint(r, c, rows[r][c])));
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
          const depth = (project(r,c,rows[r][c]).depth + project(r,c+1,rows[r][c+1]).depth + project(r+1,c+1,rows[r+1][c+1]).depth + project(r+1,c,rows[r+1][c]).depth) / 4;
          cells.push({ r, c, avg, depth });
        }
      }
      cells.sort((a, b) => a.depth - b.depth);
      for (const cell of cells) {
        const r = cell.r, c = cell.c;
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
      const kTicks = [0, Math.floor(surfaceData.strike_ratios.length / 2), surfaceData.strike_ratios.length - 1];
      kTicks.forEach((idx, i) => {
        const p = project3({ x: [-0.5, 0, 0.5][i], y: -0.58, z: 0 });
        ctx.fillText((surfaceData.strike_ratios[idx] * 100).toFixed(0) + '%', p.x - 14, p.y + 18);
      });
      const tA = project3({ x: -0.58, y: -0.5, z: 0 });
      const tB = project3({ x: -0.58, y: 0.5, z: 0 });
      ctx.fillText(Number(surfaceData.maturities[0]).toFixed(2) + 'y', tA.x - 38, tA.y + 6);
      ctx.fillText(Number(surfaceData.maturities[surfaceData.maturities.length - 1]).toFixed(2) + 'y', tB.x - 38, tB.y + 6);
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


def _bool(qs, key, default=False):
    value = str(qs.get(key, [str(default).lower()])[0]).lower()
    return value in {"1", "true", "yes", "on"}


def _choice(qs, key, default, allowed):
    value = str(qs.get(key, [default])[0]).lower()
    return value if value in allowed else default


def _validate_price_inputs(r, q, terms):
    if not np.isfinite(terms.notional) or terms.notional <= 0:
        raise ValueError("名义本金必须为正数")
    if not np.isfinite(terms.maturity_years) or terms.maturity_years <= 0:
        raise ValueError("期限必须为正数")
    if not np.isfinite(terms.coupon):
        raise ValueError("报价票息必须是有效数字")
    if not (0.0 < terms.knock_in < terms.knock_out):
        raise ValueError("敲入线必须大于 0 且低于敲出线")
    if terms.lockout_months < 0:
        raise ValueError("锁定期不能为负")
    if terms.step_down < 0:
        raise ValueError("降敲幅度不能为负")
    if not np.isfinite(r) or not (-0.1 <= r <= 0.3):
        raise ValueError("无风险利率 r 超出合理范围")
    if not np.isfinite(q) or not (-0.1 <= q <= 0.3):
        raise ValueError("分红率 q 超出合理范围")


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/price":
            try:
                self._send_json(self._price(parse_qs(parsed.query)))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
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
        uses_real_iv = False
        if index_key in ETF_OPTION_UNIVERSE:
            try:
                option_surface = load_sse_option_iv_surface(index_key, snapshot.spot)
                surface = option_surface.surface
                iv_source = option_surface.source
                iv_asof = option_surface.asof
                iv_points = f"{option_surface.filtered_points}/{option_surface.raw_points}"
                uses_real_iv = True
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
            lockout_months=_int(qs, "lockout", 0),
            knock_in_observation=_choice(qs, "ki_obs", "daily", {"daily", "maturity"}),
            knock_out_observation=_choice(qs, "ko_obs", "monthly", {"daily", "monthly", "quarterly", "maturity"}),
            step_down_enabled=_bool(qs, "step_down_enabled", False),
            step_down=_float(qs, "step_down", 0.0),
        )
        _validate_price_inputs(r, q, terms)
        paths = min(max(_int(qs, "paths", 5000), 1000), 30000)

        mc = SnowballMCPricer(lv, r, q)
        components = mc._simulate_components(snapshot.spot, terms, paths)
        par_result = mc.fair_coupon_from_components(terms, components)
        quoted_result = mc.price(snapshot.spot, terms, paths=paths, components=components)
        lv_diagnostics = lv.local_vol_diagnostics()

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
            "uses_real_iv": uses_real_iv,
            "terms": {
                "notional": terms.notional,
                "maturity_years": terms.maturity_years,
                "coupon": terms.coupon,
                "knock_out": terms.knock_out,
                "knock_in": terms.knock_in,
                "lockout_months": terms.lockout_months,
                "knock_in_observation": terms.knock_in_observation,
                "knock_out_observation": terms.knock_out_observation,
                "step_down_enabled": terms.step_down_enabled,
                "step_down": terms.step_down,
            },
            "fair_coupon": par_result["fair_coupon"],
            "unclipped_fair_coupon": par_result["unclipped_fair_coupon"],
            "fair_coupon_clipped": par_result["fair_coupon_clipped"],
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
            "atm_local_vol": lv.local_vol(snapshot.spot, min(terms.maturity_years, 1.0)),
            "local_vol_diagnostics": lv_diagnostics,
            "history": {"dates": snapshot.dates[-260:], "close": close.round(4).tolist()},
            "surface": surface_payload(surface, lv),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
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
