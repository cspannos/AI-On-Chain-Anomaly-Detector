# AI Powered On-Chain Anomaly Detector

A lightweight Python script that connects to an Ethereum (or Layer 2) node, fetches recent transactions, and flags outliers based on transaction value.

## Features

- **Fetch Recent Transactions**: Scans the latest N blocks for transactions
- **Anomaly Detection**: Uses an Isolation Forest model to identify unusually large or small ETH transfers
- **Configurable**: Easily adjust the block window size and expected anomaly rate via CLI flags or environment variables
- **Output**: Writes a timestamped JSON file listing detected anomalies for further analysis or alerting

## Usage

### Set up secrets

```bash
export ETH_RPC="https://mainnet.infura.io/v3/YOUR_KEY"
```

### Run the script

```bash
python scripts/detect_anomalies.py \
  --window 1000 \
  --contamination 0.01 \
  --out data/anomalies.json
```

### Review results

Anomalies are saved in `data/anomalies.json` with timestamps.

## Requirements

- **Python 3.7+**
- **Dependencies**: `pandas`, `web3`, `scikit-learn`

## Integration

Designed to run via GitHub Actions every n minutes. On each run, it updates the anomalies log, which can trigger alerts or power dashboards.
