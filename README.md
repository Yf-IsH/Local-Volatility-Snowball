# Local Volatility Snowball Coupon Demo

This project demonstrates how a local volatility model can be used to quote the fair annual coupon of a snowball/autocallable product.

## What It Does

- Fetches daily market data from Eastmoney.
- Uses SSE ETF option risk indicators (`IMPLC_VOLATLTY`) to build a real ETF option IV surface.
- Converts the IV surface into a local volatility surface with the Dupire formula.
- Simulates snowball paths with local volatility Monte Carlo.
- Solves the fair annual coupon from `PV(coupon) = notional`.
- Shows an interactive pricing workbench with IV/local-vol surfaces, path outcome probabilities, model diagnostics, cash-flow details, and a lecture-style math derivation in the browser.

## Recent Updates

- Corrected SSE ETF option expiry parsing from "last Wednesday" to the official "fourth Wednesday of the expiry month" convention.
- Added Dupire diagnostics for positive density, positive numerator, fallback ratio, and local-volatility range.
- Improved Monte Carlo pricing by reusing simulated payoff components for both fair coupon and quoted coupon valuation.
- Added unclipped fair coupon output so extreme or invalid quotes are visible instead of silently hidden by display clipping.
- Reworked the browser demo into a more usable pricing workbench with grouped inputs, result cards, risk probabilities, market data, surface views, detailed cash-flow components, and model notes.
- Expanded the model explanation tab into a longer lecture-style derivation without relying on external MathJax/CDN resources.
- Added a detailed mathematical review document at `docs/local_vol_pricer_math_review.tex`.
- Added regression tests for Dupire diagnostics, SSE ETF expiry parsing, and fair-coupon consistency.

## Environment Setup

Do not assume a `.venv` folder already exists. `.venv` is only a local virtual environment directory on one machine, and it should not be committed to Git.

Choose one of the following setup methods.

### Option A: Python venv

Create and activate a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the demo:

```powershell
python demo_app.py --port 8020
```

Run tests:

```powershell
python -m pytest tests -v
```

### Option B: Conda

Create and activate a Conda environment:

```powershell
conda create -n local-vol-pricer python=3.11
conda activate local-vol-pricer
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the demo:

```powershell
python demo_app.py --port 8020
```

Run tests:

```powershell
python -m pytest tests -v
```

## Open The App

After starting the server, open:

```text
http://127.0.0.1:8020
```

If the port is already in use, the app automatically tries the next available port and prints the actual URL in the terminal.

## Data Notes

For ETF underlyings, option IV comes from the SSE option risk indicator field `IMPLC_VOLATLTY`. The app filters zero or extreme IV values and interpolates the remaining contract points into a regular IV surface.

SSE ETF option expiry dates are parsed using the fourth Wednesday of the expiry month. If a production deployment needs full holiday handling, the next step is to connect an exchange trading calendar and roll holidays or market-closure days according to the official rule.

For broad indices such as CSI 500 and CSI 1000, the app still uses a demo IV surface generated from historical realized volatility. A production-grade version should connect to CFFEX index option chains and imply IV from option prices.

Runtime market data cache files are stored under `data/cache/` and are ignored by Git.

## Model Notes

The pricing engine treats option prices as the theoretical market object, while the app often receives or displays the same information in implied-volatility coordinates. In definition, implied volatility is obtained by solving the Black-Scholes equation from a market option price. In implementation, the app maps the available implied-volatility surface back into equivalent discounted Black-Scholes call prices, then applies the Dupire relation to infer a local volatility surface:

```text
sigma_local^2(K,T) = (dC/dT + q C + (r - q) K dC/dK) / (0.5 K^2 d2C/dK2)
```

The snowball payoff is path dependent, so the final coupon valuation is handled by Monte Carlo simulation under the local-volatility dynamics:

```text
dS_t / S_t = (r - q) dt + sigma_local(S_t,t) dW_t
```

The app reports both the fair annual coupon and the quoted-coupon present value. It also reports knock-out, knock-in, knock-in-without-knock-out, and no-knock-in/no-knock-out probabilities so the result can be reviewed beyond a single coupon number.

## Validation

Run the test suite with:

```powershell
python -m pytest tests -q
```

Current regression coverage checks:

- calibration sanity for the IV surface,
- Dupire diagnostics on a flat volatility surface,
- SSE ETF fourth-Wednesday expiry parsing,
- fair-coupon consistency when the solved coupon is priced back into the product.
