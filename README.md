#  GridWise

GridWise is a prototype of a community-driven, AI-optimized smart grid that enables households to share, trade, and optimize energy in real time. Powered by Agentic AI and IBM WatsonX, it learns from usage patterns, predicts surplus/deficit, and proactively suggests transparent peer-to-peer energy trades — ensuring fairness, efficiency, and sustainability.

---

##  Key Features  

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
├── chroma_db/                # Chroma vector database files
├── faiss_db/                 # FAISS indices
│   ├── energy/               # Energy-specific FAISS index
│   └── patterns/             # Usage pattern FAISS index
├── sample_data/              # Datasets for testing
│   ├── bangalore_energy_consumption.csv
│   └── load_history1.csv
├── templates/                # HTML templates for frontend
│   └── index.html
├── flask_app.py              # Flask-based server for running GridWise
├── mcp_Server.py             # Core backend logic
├── requirements.txt          # Python dependencies
├── constraints.txt           # Constraint definitions
├── index.html                # Standalone landing page
├── .env                      # Environment variables (WatsonX configs, API keys)
└── README.md                 # Project documentation

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

##  How to Run

Start the agent server:
```bash
python flask_app.py
```
view the UI on 
```bash
http://127.0.0.1:5000/
```
##  Core Concepts

### The intelligence of this app comes from its RAG pipeline:

#### Ingestion – RAGPipeline reads historical energy data (CSV → Document objects → Embeddings).

#### Vectorstore – Embeddings stored in FAISS, supporting:

#### Raw energy consumption data

#### Aggregated daily patterns

#### Retrieval – Queries retrieve similar historical data and patterns.

#### Generation – Retrieved context + user request → sent to IBM watsonx.ai LLM, which produces grounded forecasts, analyses, or actionable recommendations.
