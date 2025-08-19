import json
import os
import sys
import argparse
import datetime
import pandas as pd
from web3 import Web3
from sklearn.ensemble import IsolationForest

def fetch_recent_transactions(w3: Web3, window: int) -> pd.DataFrame:
    """Fetch transactions from the latest `window` blocks."""
    latest = w3.eth.block_number
    tx_data = []
    start_block = max(latest - window, 0)
    for block_num in range(start_block, latest + 1):
        block = w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            tx_data.append({
                "block": block_num,
                "value": float(w3.from_wei(tx.value, 'ether')),
            })
    return pd.DataFrame(tx_data)

def detect_anomalies(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Flag outliers in the `value` feature using IsolationForest."""
    model = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df['anomaly_flag'] = model.fit_predict(df[['value']])
    return df[df['anomaly_flag'] == -1].drop(columns=['anomaly_flag'])

def main():
    parser = argparse.ArgumentParser(description="On-chain anomaly detector.")
    parser.add_argument("--window", type=int, default=int(os.getenv('WINDOW', 50)),
                        help="Number of blocks to scan (default from WINDOW env).")
    parser.add_argument("--contamination", type=float, default=float(os.getenv('CONTAMINATION', 0.01)),
                        help="Expected anomaly rate (default from CONTAMINATION env).")
    parser.add_argument("--out", type=str, default=os.getenv('OUT_PATH', 'data/anomalies.json'),
                        help="Output JSON path.")
    args = parser.parse_args()

    rpc_url = os.getenv("ETH_RPC")
    if not rpc_url:
        print("Error: ETH_RPC environment variable is not set.")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"Error: Cannot connect to Ethereum node at {rpc_url}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = fetch_recent_transactions(w3, args.window)
    anomalies = detect_anomalies(df, args.contamination)

    output = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "anomalies": anomalies.to_dict(orient='records')
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(output['anomalies'])} anomalies to {args.out}")

if __name__ == "__main__":
    main()
