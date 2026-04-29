# DDoS Detection using Graph Attention Networks (GAT)

This project implements a state-of-the-art Deep Learning approach to detect Distributed Denial of Service (DDoS) attacks using **Graph Neural Networks (GNNs)**. Specifically, it utilizes a **Graph Attention Network (GAT)** to model network traffic as a graph, allowing it to capture complex relational patterns between IP addresses that traditional machine learning methods often miss.

## 🚀 Key Features
- **Graph-Based Modeling**: Converts standard network flow data (CSV) into a graph where IPs are nodes and flows are edges.
- **Attention Mechanism**: Uses Multi-Head Attention to identify which neighbors in the network contribute most to malicious patterns.
- **Live Traffic Inference**: Includes a real-time sniffer using Scapy to capture live packets and perform instant DDoS detection.
- **Web Dashboard**: An interactive Flask-based interface to visualize the network graph, monitor traffic, and view detection results.
- **Automated Pipeline**: End-to-end scripts for data cleaning, feature scaling, graph building, training, and evaluation.

## 🏗️ Architecture
The system follows a **Research -> Strategy -> Execution** workflow:
1. **Data Ingestion**: Processes CIC-IDS2019 datasets.
2. **Graph Construction**: Aggregates flow features into node features and establishes topological connections.
3. **GAT Model**: A two-layer Graph Attention Network that learns to classify nodes (IPs) as "Benign" or "Attack".
4. **Real-time Monitoring**: A background sniffer that slides a time window over live traffic to detect ongoing attacks.

## 📁 Project Structure
- `app.py`: The main Flask application for the web dashboard and live detection.
- `main.py`: The core training pipeline for offline model development.
- `src/`:
  - `model.py`: PyTorch Geometric implementation of the GAT architecture.
  - `graph_builder.py`: Logic for transforming tabular data into PyG Graph objects.
  - `live_capture.py`: Real-time packet capture and flow aggregation.
  - `flow_tracker.py`: State management for active network connections.
- `config.py`: Centralized configuration for hyperparameters and network settings.
- `data/raw/`: Directory for placing dataset CSV files.
- `results/`: Stores trained models (`best_model.pt`), scalers, and performance metrics.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Npcap (for Windows live capture) or Libpcap (for Linux).

### Quick Setup (Windows)
1. Run `setup.bat`. This will:
   - Create a virtual environment (`.venv`).
   - Install all dependencies (PyTorch, PyG, Scapy, Flask, etc.).
   - Create a Desktop shortcut for easy access.

## 🖥️ Usage

### 1. Training the Model
To train the model on a specific dataset:
1. Place your CSV file in `data/raw/`.
2. Run `python main.py` or use the training interface in the web app.

### 2. Running the Live Detector
1. Execute `run.bat` or run `python run.py`.
2. Open the web dashboard (typically `http://127.0.0.1:5000`).
3. Toggle "Live Capture" to start monitoring your network interface.

## 📊 Why GNN for DDoS?
Traditional Machine Learning (like Random Forests or SVMs) analyzes packets in isolation. DDoS attacks, however, are **distributed** by nature. A GNN looks at the **graph topology**:
- It identifies coordinated behavior from multiple source IPs.
- It sees "hubs" of traffic converging on a single target.
- The **Attention Mechanism** allows the model to ignore noisy benign traffic and focus on the high-impact malicious edges.

## 📜 License
This project is developed for educational and research purposes in the field of cybersecurity and graph machine learning.
