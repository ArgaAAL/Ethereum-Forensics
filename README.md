# 🕵️‍♂️ Ethereum Forensics & Analytics Suite

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=ArgaAAL.Ethereum-Forensics)

> **A professional Ethereum analytics and forensic framework for account-based blockchain monitoring. It reconstructs transaction graphs, profiles gas usage, internal calls, and token interactions.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Etherscan API](https://img.shields.io/badge/RPC-Etherscan-blue)](https://etherscan.io/apis)
[![XGBoost](https://img.shields.io/badge/AI-XGBoost-green)](https://xgboost.readthedocs.io/)
[![ONNX](https://img.shields.io/badge/Deploy-ONNX-purple)](https://onnx.ai/)

---

## ⚡ Executive Summary

Ethereum forensic analysis is challenging due to account-based abstraction and smart contract complexity. This framework:

* **Profiles address behavior**: contract interactions, ERC20/ERC721 activity, cross-chain bridge usage.  
* **Tracks rapid fund movements** and internal message calls.  
* **Implements ML detection** for anomalous or suspicious patterns.  

---

## 🏗️ System Architecture

```mermaid
graph TD
    RPC[Etherscan/Infura RPC] -->|Raw Blocks| Ingest[Ingestion Engine]
    Price[CryptoCompare Oracle] -->|Spot Rates| Ingest
    
    subgraph "Layer 1: Data Decomposition"
        Ingest --> ERC20[ERC20 Token Parser]
        Ingest --> Internal[Internal Tx Tracer]
        Ingest --> Gas[Gas & Nonce Profiler]
    end
    
    subgraph "Layer 2: Feature Synthesis"
        ERC20 & Internal & Gas --> Vector[Feature Vector Builder]
    end
    
    subgraph "Layer 3: Classification"
        Vector --> XGB[XGBoost Classifier]
        Vector --> ONNX[ONNX Runtime (Production)]
    end
    
    XGB --> Risk["Risk Score (0.0 - 1.0)"]
```

---

## 🧩 Core Modules

### 1\. ETL Pipeline (`src/core/etl_pipeline.py`)
Normalizes raw EVM data, parses internal messages, and profiles ERC20/ERC721 token transfers.

### 2\. Token Analyzer (`src/core/token_analyzer.py`)
Analyzes contract calls, approvals, and liquidity events to detect suspicious patterns.

### 3\. ML Models (`src/models/xgboost_trainer.py` & `src/models/onnx_exporter.py`)
Gradient Boosting model trained on known on-chain patterns; ONNX export enables low-latency deployment.

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/ArgaAAL/Ethereum-Forensics.git
cd Ethereum-Forensics
pip install -r requirements.txt
cp .env.example .env
# Add your ETHERSCAN_API_KEY and CRYPTOCOMPARE_API_KEY
```

---

## 🚀 Usage

Run the ETL pipeline for an address:

```bash
python src/core/etl_pipeline.py --address 0x123...
```

Retrain the model:

```bash
python src/models/xgboost_trainer.py --data new_labels.csv
```

---

## 📜 License

MIT License.

*Part of the **Crypto-Threat-Intelligence** suite.*
