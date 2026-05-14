# Local Volatility Snowball Coupon Demo

This project demonstrates how a local volatility model can be used to quote the fair annual coupon of a snowball/autocallable product.

## What It Does

- Fetches daily market data from Eastmoney.
- Uses SSE ETF option risk indicators (`IMPLC_VOLATLTY`) to build a real ETF option IV surface.
- Converts the IV surface into a local volatility surface with the Dupire formula.
- Simulates snowball paths with local volatility Monte Carlo.
- Solves the fair annual coupon from `PV(coupon) = notional`.
- Shows interactive 3D IV/local-vol surfaces, path outcome probabilities, and a detailed math derivation in the browser.

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
python demo_app.py --port 8015
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
python demo_app.py --port 8015
```

Run tests:

```powershell
python -m pytest tests -v
```

## Open The App

After starting the server, open:

```text
http://127.0.0.1:8015
```

If the port is already in use, the app automatically tries the next available port and prints the actual URL in the terminal.

## Data Notes

For ETF underlyings, option IV comes from the SSE option risk indicator field `IMPLC_VOLATLTY`. The app filters zero or extreme IV values and interpolates the remaining contract points into a regular IV surface.

For broad indices such as CSI 500 and CSI 1000, the app still uses a demo IV surface generated from historical realized volatility. A production-grade version should connect to CFFEX index option chains and imply IV from option prices.

Runtime market data cache files are stored under `data/cache/` and are ignored by Git.
