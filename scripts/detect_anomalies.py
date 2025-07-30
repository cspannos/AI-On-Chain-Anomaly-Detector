import json
from web3 import Web3
import pandas as pd
from sklearn.ensemble import IsolationForest
import datetime

def fetch_recent_transactions(w3: Web3, window: int):
    # Fetch the latest `window` blocks of transactions
    latest = w3.eth.block_number
    tx_data = []
    for block_num in range(latest - window, latest + 1):
        block = w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            tx_data.append({
                "block": block_num,
                "value": w3.from_wei(tx.value, 'ether'),
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
    RPC_URL = "${{ secrets.ETH_RPC }}"
    WINDOW = 1000          # number of blocks to analyze per run
    CONTAMINATION = 0.01   # fraction of data expected to be anomalies

    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.isConnected():
        raise ConnectionError("Failed to connect to Ethereum node")

    # Fetch and analyze
    df = fetch_recent_transactions(w3, WINDOW)
    anomalies = detect_anomalies(df, CONTAMINATION)

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
