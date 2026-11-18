# Binance BTC Futures Strategy Backtester (both type consisting of - NVIDIA GPU (CUDA) & CPU based Multiprocessing Accelerated)

> Built on - March 2025,
> This repository is a collection of multiple high-performance Python backtesting strategies framework built to evaluate, optimize, and scan massive parameter spaces for quantitative crypto trading strategies on 1-minute Binance Futures OHLC data using NVIDIA'S GPU through CUDA

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![CUDA/CuPy](https://img.shields.io/badge/Accelerated%20by-NVIDIA%20CUDA%20%2F%20CuPy-green.svg)](https://cupy.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [About The Repository](#-about-the-repository)
- [Key Features & Evolution](#-key-features--evolution)
- [Project Scripts Overview](#-project-scripts-overview)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 About The Repository

This repository contains an evolving suite of backtesting scripts designed to optimize trading strategies (primarily moving average crossovers, ATR filters, trailing stop losses, and custom gap deviations) using historical 1-minute Binance BTC Futures data. 

Because grid-searching millions of parameter variations sequentially takes hours or days on a standard CPU, this project utilizes **NVIDIA CUDA (via CuPy custom raw kernels)** and **Python Multiprocessing** to push execution speeds down to seconds or minutes, fully maximizing GPU multiprocessors.

---

## ✨ Key Features & Evolution

* **CPU Multiprocessing (`backtester.py`):** Splits heavy parameter loops across multiple CPU cores using Python's `multiprocessing` library.
* **GPU Acceleration (`cuda backtesting v5...py`):** Offloads loop execution and metric tracking to NVIDIA GPUs using `CuPy` arrays and `joblib` caching.
* **Custom CUDA Raw Kernels (`cuda v6` - `v9`):** Implements low-level C-style CUDA kernels (`cp.RawKernel`) to run hundreds of thousands of parameter sets completely in parallel across GPU threads and blocks.
* **Advanced Risk Management Filters:** Includes logic for Average True Range (ATR) thresholds, dynamic EMA deviations, and trailing stop-loss (TSL) mechanics.
* **Automated Post-Processing:** Automatically extracts top-performing parameter permutations, sorts results by Net Profit or Win Percentage, and exports reports into CSV or Excel formats.

---

## 📂 Project Scripts Overview

The repository features several experimental versions showcasing the progression from CPU-bound multiprocessing to high-speed GPU kernel execution:

| Script Filename | Description |
| :--- | :--- |
| `backtester.py` | CPU-based multiprocessing backtester using standard `pandas` and `pandas_ta`. |
| `cuda backtesting v5 parallel gpu multiprocessing.py` | Early GPU integration using `joblib` and `mp.Pool` chunking across CUDA cores. |
| `cuda backtesting v6 assigning the free GPU threads in cuda cores.py` | Introduction of custom CUDA Raw Kernels with trailing stop loss (TSL) parameters. |
| `cuda v7 bakctesting and post processing on gpu.py` | Optimized block/thread grid mappings capable of running millions of parameter permutations efficiently. |
| `cuda v8 WITHOUT ema gap.py` | Variant focusing on streamlined EMA trend conditions without complex gap calculations, exporting top net profit/win rate metrics. |
| `cuda v9.py` / `v9 incorrect output.py` | Refined kernel builds incorporating custom mid-price calculations (`hl2`), strict deviation boundaries, and rapid CSV generation. |
| `ema difference checker.py` | Utility script to inspect and sort 34-period EMA delta distributions across candle sequences. |

---

## ⚙️ Prerequisites

Make sure your environment meets the following specifications:
* **Operating System:** Windows 10/11 or Linux (with NVIDIA drivers installed)
* **Hardware:** An NVIDIA GPU with CUDA capability (e.g., GTX/RTX series)
* **Python:** Version 3.8 or higher
* **CUDA Toolkit:** Matching your installed CuPy version

---

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/pandiyan07/1min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git](https://github.com/pandiyan07/1min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git)
cd your-repo-name
```

2. **Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```


3. **Install dependencies:**
Ensure you install the correct CuPy wheel corresponding to your local CUDA version (e.g., `cupy-cuda12x`):
```bash
pip install pandas numpy pandas_ta cupy-cuda12x joblib openpyxl

```



---

## 💡 Usage

1. Place your historical OHLC data file (e.g., `data_ohlc_feb1.csv`) in the root directory or update the path variable inside the target script.
2. Run your preferred backtesting script via command line:
```bash
python "cuda v9.py"

```


3. Check the output directory for generated CSV or Excel performance reports (e.g., `net_profit.csv`, `win_percentage.csv`).

---

## 🗂️ Project Structure

```text
your-repo-name/
│
├── cache/                  # Joblib function caching directory
├── data_ohlc_feb1.csv      # Sample 1-min Binance Futures OHLC dataset
├── backtester.py           # CPU multiprocessing baseline
├── cuda v7...py            # GPU kernel implementations
├── cuda v9.py              # Latest optimized GPU backtester
├── ema difference checker.py # Indicator utility script
├── .gitignore
└── README.md

```

---

## 🤝 Contributing

Contributions, feature requests, and optimizations are welcome! Please feel free to open an issue or submit a pull request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
