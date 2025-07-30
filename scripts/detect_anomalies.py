import json
from web3 import Web3
import pandas as pd
from sklearn.ensemble import IsolationForest
import datetime
import os

def fetch_recent_transactions(w3: Web3, window: int):
    # Fetch the latest `window` blocks of transactions
    latest = w3.eth.block_number
    tx_data = []
    start_block = max(latest - window, 0)
    for block_num in range(start_block, latest + 1):
        block = w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            tx_data.append({
                "block": block_num,
                "value": float(w3.from_wei(tx.value, 'ether')),
                # add more features if desired
            })
    return pd.DataFrame(tx_data)

def detect_anomalies(df: pd.DataFrame, contamination: float):
    model = IsolationForest(contamination=contamination, random_state=42)
    features = df[["value"]]
    df['anomaly'] = model.fit_predict(features)
    anomalies = df[df['anomaly'] == -1]
    return anomalies.drop(columns=['anomaly'])

def main():
    # Configuration
    RPC_URL = os.getenv("ETH_RPC")
    WINDOW = int(os.getenv("WINDOW", 1000))          # number of blocks to analyze per run
    CONTAMINATION = float(os.getenv("CONTAMINATION", 0.01))   # fraction expected anomalies

    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to Ethereum node at {}".format(RPC_URL))

    # Fetch and analyze
    df = fetch_recent_transactions(w3, WINDOW)
    anomalies = detect_anomalies(df, CONTAMINATION)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Output to JSON
    timestamp = datetime.datetime.utcnow().isoformat()
    output = {
        "timestamp": timestamp,
        "anomalies": anomalies.to_dict(orient='records')
    }
    with open("data/anomalies.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
