# ⚡ Grid Operations AI Assistant  

An AI-powered assistant for managing **grid operations**, including electricity consumption forecasting, capacity planning, generation monitoring, and load optimization.  
It leverages a **Retrieval-Augmented Generation (RAG)** pipeline to provide **grounded, data-driven insights** from historical energy consumption records.  

---

## 🔑 Key Features  

- **Load Forecasting** – Predict future electricity demand for specific cities or areas based on historical data.  
- **Capacity Planning** – Calculate the necessary prepared energy capacity for the next 3 hours, considering a reserve margin and historical consumption patterns.  
- **Generation Monitoring** – Analyze real-time generation data from distributed sources and automatically flag anomalies or shortfalls.  
- **Load Optimization** – Recommend concrete, sector-specific actions to balance supply and demand and prevent power outages.  
- **Consumption Analysis** – Dive deep into historical data to identify key consumption trends, peak hours, and low-usage periods.  
- **Historical Similarity Search** – Find days with similar consumption patterns to a target date, helping operators understand past behavior under comparable conditions.  

---

## 🛠️ Technology Stack  

- **Framework**: [FastMCP](https://github.com/) (based on FastAPI) for building the agent’s server.  
- **AI/ML**:  
  - [IBM watsonx.ai](https://www.ibm.com/watsonx/ai) – Core LLM for reasoning, analysis, and generation.  
  - [LangChain](https://www.langchain.com/) – Orchestrates the RAG pipeline.  
  - [FAISS](https://github.com/facebookresearch/faiss) – Vector database for similarity search.  
  - [Sentence-Transformers](https://www.sbert.net/) – Generates embeddings for historical data.  
- **Data Handling**: [pandas](https://pandas.pydata.org/) for manipulation, [matplotlib](https://matplotlib.org/) for visualizations.  
- **Utilities**: [python-dotenv](https://github.com/theskumar/python-dotenv) for environment variable management.  

---

## ⚙️ Installation and Setup  

### 📂 Project Structure
```bash
├── sample_data/
│   └── load_history1.csv   # historical energy data
├── flask_app.py            # main agent server
├── rag_pipeline.py         # RAG pipeline logic
├── requirements.txt        # dependencies
├── constraints.txt         # dependency constraints
└── README.md               # this file
```

### 1. Clone the repository  
```bash
git clone <repository_url>
cd <repository_folder>
```
### 2. Install dependencies
This project uses a constraints file (constraints.txt) for consistent dependency resolution.
```bash
pip install -r requirements.txt -c constraints.txt --upgrade
```
### 3. Configure API credentials
Create a .env file in the project root:
```ini
# .env
WATSONX_API_KEY="<your_api_key>"
WATSONX_PROJECT_ID="<your_project_id>"
WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Or your specific endpoint
```
### 4. Prepare data
Place your historical energy consumption data in:
```bash
sample_data/load_history1.csv
```

## 🚀 How to Run

Start the agent server:
```bash
python flask_app.py
```
view the UI on 
```bash
http://127.0.0.1:5000/
```
## 🧠 Core Concepts

### The intelligence of this app comes from its RAG pipeline:

#### Ingestion – RAGPipeline reads historical energy data (CSV → Document objects → Embeddings).

#### Vectorstore – Embeddings stored in FAISS, supporting:

#### Raw energy consumption data

#### Aggregated daily patterns

#### Retrieval – Queries retrieve similar historical data and patterns.

#### Generation – Retrieved context + user request → sent to IBM watsonx.ai LLM, which produces grounded forecasts, analyses, or actionable recommendations.
