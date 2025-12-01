# 🕵️‍♂️ Ethereum Forensics: Advanced EVM Analysis Engine

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=ArgaAAL.Ethereum-Forensics)

> **A forensic-grade analytics pipeline designed to characterize, trace, and classify transactional behavior across the Ethereum Execution Layer.**  
> Uses multi-source ETL, internal-call decomposition, live pricing, and ML-based heuristics to identify anomalous behavior across EOAs, smart contracts, and routing contracts.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Etherscan API](https://img.shields.io/badge/RPC-Etherscan-blue)](https://etherscan.io/apis)
[![ONNX](https://img.shields.io/badge/Runtime-ONNX-purple)](https://onnx.ai/)

---

## ⚡ Overview

Ethereum’s account-based model introduces unique forensic challenges:

- Contract mediation obscures ownership trails.
- Internal transactions (CALL/DELEGATECALL) enable deeply nested fund routing.
- On-chain liquidity protocols allow instant asset liquidation.
- Gas markets reveal intent but are non-linear and highly contextual.

This engine consolidates these complexities into a cohesive forensic workflow.  
The system is optimized for **high-volume address profiling**, **model retraining**, and **graph-aware behavioral analysis**.

---

## 🏗️ Architecture

```mermaid
graph TD
    RPC[Etherscan RPC] -->|Raw Blocks / Transactions| Ingest[Ingestion Engine]
    Oracle[CryptoCompare Oracle] -->|USD Valuations| Ingest
    
    subgraph L1 [Layer 1: Data Decomposition]
        Ingest --> ERC20[ERC20 Token Analyzer]
        Ingest --> InternalTx[Internal Call Tracer]
        Ingest --> GasProfiler[Gas & Nonce Profiler]
    end
    
    subgraph L2 [Layer 2: Feature Encoding]
        ERC20 --> FeatureBuilder
        InternalTx --> FeatureBuilder
        GasProfiler --> FeatureBuilder
    end

    FeatureBuilder --> XGB[XGBoost Model]
    FeatureBuilder --> ONNXRuntime[ONNX Runtime]

    XGB --> Score["Risk Score (0.0–1.0)"]
    ONNXRuntime --> Score
````

---

## 🧩 Components

### **1. ETL Engine — `src/core/etl_pipeline.py`**

Handles RPC ingestion, aggregation, and normalization.

Key Abilities:

* Differentiates EOAs, proxies, routers, and vault contracts.
* Extracts and resolves internal message calls.
* Identifies gas spikes, frontrunning signatures, and automated routing patterns.
* Reconstructs chronological transfer flows using a temporal graph.

---

### **2. Token Analysis Engine — `src/core/token_analyzer.py`**

Specialized for ERC-20/ERC-721 patterns.

Detects:

* Approval surges (common pre-drain signatures).
* LP token mint/burn activity.
* Concentrated token exits via AMMs (indicative of liquidation phases).
* “Hidden transfers” routed through wrappers and nested proxy calls.

---

### **3. Internal Transaction Tracer — `src/utils/data_processor.py`**

Interprets CALL, STATICCALL, and DELEGATECALL chains.

Capabilities:

* Expands the transaction execution tree.
* Normalizes nested transfers into human-readable flows.
* Captures delegated execution where the parent address masks the true operator.

---

### **4. ML Model + Deployment — `src/models/xgboost_trainer.py`**

Gradient-boosted classifier for behavior scoring.

Features:

* Time-based activity patterns
* Contract interaction entropy
* Token flow irregularity
* Gas/nonce distribution variance
* Deviation from normative account behavior

Exports to ONNX for real-time deployment (<10ms inference).

---

## 🛠️ Setup

```bash
git clone https://github.com/ArgaAAL/Ethereum-Forensics.git
cd Ethereum-Forensics

pip install -r requirements.txt

cp .env.example .env
# Add your API keys
```

---

## 🚀 Usage

### **Profile an Address**

```bash
python src/core/etl_pipeline.py --address 0xABC...
```

Generated output includes:

* Behavioral descriptors
* ERC-20/721 asset flows
* Nested internal transfer graph
* Synthetic risk score
* Time-weighted USD valuations

---

### **Retrain the Model**

```bash
python src/models/xgboost_trainer.py --data training_dataset.csv
```

---

## 📜 License

MIT License.

---

## 📚 Additional Notes

This project is part of a broader research effort into **EVM-based transactional intelligence**, focusing on:

* Temporal behavior modeling
* Contract interaction profiling
* Dynamic fund-flow reconstruction
* Low-latency risk classification for streaming pipelines
* Multi-chain extension pathways (planned)

The repository structure, documentation style, and ML export format are optimized for security teams, quant researchers, and forensic analysts working with high-throughput on-chain data.

---
