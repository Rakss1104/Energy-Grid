import os, json, uuid, tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# RAG pipeline imports
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging

from pydantic import BaseModel, Field, conlist

# --- MCP (Model Context Protocol) server ---
from mcp.server.fastmcp import FastMCP

# --- watsonx.ai SDK ---
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# ---------------------------
# Load config
# ---------------------------
load_dotenv()
API_KEY = os.environ["WATSONX_API_KEY"]
BASE_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]
MODEL_ID = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

creds = Credentials(api_key=API_KEY, url=BASE_URL)
llm = ModelInference(model_id=MODEL_ID, credentials=creds, project_id=PROJECT_ID)

app = FastMCP("grid-ops")
path_to_csv="sample_data/load_history1.csv"
# ---------------------------
# RAG Pipeline Setup
# ---------------------------
class RAGPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collections for different data types
        self.energy_collection = self.client.get_or_create_collection(
            name="energy_consumption",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        self.patterns_collection = self.client.get_or_create_collection(
            name="consumption_patterns",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def ingest_energy_data(self, df: pd.DataFrame):
        """Ingest energy consumption data into ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        # Group by date and area for better context
        for _, row in df.iterrows():
            doc_text = f"""
            Date: {row['Date']} {row['Hour']}
            Location: {row['Area']}, {row['City']}
            Energy Consumption:
            - Industrial: {row['Industrial']} MW
            - Households: {row['Households']} MW
            - Schools: {row['Schools']} MW
            - Colleges: {row['Colleges']} MW
            - Hospitals: {row['Hospitals']} MW
            - Total: {row['Total']} MW
            """
            
            documents.append(doc_text)
            metadatas.append({
                "date": row['Date'],
                "hour": row['Hour'],
                "city": row['City'],
                "area": row['Area'],
                "total_mw": float(row['Total']),
                "industrial": float(row['Industrial']),
                "households": float(row['Households']),
                "schools": float(row['Schools']),
                "colleges": float(row['Colleges']),
                "hospitals": float(row['Hospitals'])
            })
            ids.append(f"{row['Date']}_{row['Hour']}_{row['City']}_{row['Area']}")
        
        # Add documents to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.energy_collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        
        self.logger.info(f"Ingested {len(documents)} energy consumption records")

    def ingest_patterns(self, df: pd.DataFrame):
        """Extract and ingest consumption patterns"""
        # Create pattern summaries by area and time
        patterns = []
        
        # Daily patterns by area
        for area in df['Area'].unique():
            area_data = df[df['Area'] == area]
            daily_pattern = area_data.groupby('Hour')[['Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals', 'Total']].mean()
            
            pattern_text = f"""
            Area: {area} Daily Consumption Pattern
            Peak Hours: {daily_pattern['Total'].idxmax()} (Peak: {daily_pattern['Total'].max():.2f} MW)
            Low Hours: {daily_pattern['Total'].idxmin()} (Low: {daily_pattern['Total'].min():.2f} MW)
            Average Industrial: {daily_pattern['Industrial'].mean():.2f} MW
            Average Residential: {daily_pattern['Households'].mean():.2f} MW
            Average Schools: {daily_pattern['Schools'].mean():.2f} MW
            Average Colleges: {daily_pattern['Colleges'].mean():.2f} MW
            Average Hospitals: {daily_pattern['Hospitals'].mean():.2f} MW
            """
            
            patterns.append({
                "text": pattern_text,
                "metadata": {
                    "area": area,
                    "pattern_type": "daily",
                    "peak_hour": daily_pattern['Total'].idxmax(),
                    "peak_mw": float(daily_pattern['Total'].max()),
                    "low_hour": daily_pattern['Total'].idxmin(),
                    "low_mw": float(daily_pattern['Total'].min()),
                    "avg_total": float(daily_pattern['Total'].mean())
                },
                "id": f"daily_pattern_{area}"
            })
        
        # Add patterns to collection
        if patterns:
            self.patterns_collection.add(
                documents=[p["text"] for p in patterns],
                metadatas=[p["metadata"] for p in patterns],
                ids=[p["id"] for p in patterns]
            )
        
        self.logger.info(f"Ingested {len(patterns)} consumption patterns")

    def query_similar_data(self, query: str, collection_name: str = "energy_consumption", n_results: int = 10):
        """Query similar data from ChromaDB"""
        collection = self.energy_collection if collection_name == "energy_consumption" else self.patterns_collection
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results

    def get_historical_context(self, city: str, area: str = None, hours_back: int = 168):
        """Get historical context for forecasting"""
        where_filter = {"city": city}
        if area:
            where_filter["area"] = area
            
        results = self.energy_collection.query(
            query_texts=[f"energy consumption in {city} {area or ''}"],
            n_results=hours_back,
            where=where_filter
        )
        
        return results

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# ---------------------------
# Data loading and ingestion
# ---------------------------
def load_and_ingest_data(csv_path: str=path_to_csv):
    """Load CSV data and ingest into RAG pipeline if not already embedded"""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using sample data generation.")
        return generate_sample_data()
    
    df = pd.read_csv(csv_path)
    
    # ✅ Only embed if collection is empty
    if rag_pipeline.energy_collection.count() == 0:
        print("⚡ No embeddings found. Ingesting energy data...")
        rag_pipeline.ingest_energy_data(df)
        rag_pipeline.ingest_patterns(df)
    else:
        print("✅ Embeddings already exist in ChromaDB. Skipping ingestion.")
    
    return df


def generate_sample_data():
    """Generate sample data matching the new format"""
    cities = ["Bangalore"]
    areas = ["MG Road", "Brigade Road", "Church Street", "Richmond Town", 
             "Shivajinagar", "Vasanth Nagar", "Ulsoor (Halasuru)", 
             "Indiranagar", "Koramangala", "Malleshwaram", "Rajajinagar"]
    
    data = []
    start_date = datetime(2025, 1, 1)
    
    for day in range(30):  # 30 days of data
        current_date = start_date + timedelta(days=day)
        for hour in range(24):
            for city in cities:
                for area in areas:
                    # Generate realistic consumption patterns
                    base_industrial = np.random.uniform(50, 90)
                    base_households = np.random.uniform(30, 50)
                    base_schools = np.random.uniform(1, 3) if 8 <= hour <= 16 else np.random.uniform(0.5, 1.5)
                    base_colleges = np.random.uniform(2, 5) if 9 <= hour <= 17 else np.random.uniform(0.5, 2)
                    base_hospitals = np.random.uniform(15, 30)
                    
                    # Add time-of-day patterns
                    if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak hours
                        multiplier = 1.2
                    elif 23 <= hour <= 5:  # Low hours
                        multiplier = 0.7
                    else:
                        multiplier = 1.0
                    
                    industrial = base_industrial * multiplier
                    households = base_households * multiplier
                    schools = base_schools
                    colleges = base_colleges
                    hospitals = base_hospitals
                    total = industrial + households + schools + colleges + hospitals
                    
                    data.append({
                        "Date": current_date.strftime("%Y-%m-%d"),
                        "Hour": f"{hour:02d}:00",
                        "City": city,
                        "Area": area,
                        "Industrial": round(industrial, 2),
                        "Households": round(households, 2),
                        "Schools": round(schools, 2),
                        "Colleges": round(colleges, 2),
                        "Hospitals": round(hospitals, 2),
                        "Total": round(total, 2)
                    })
    
    df = pd.DataFrame(data)
    df.to_csv("sample_energy_data.csv", index=False)

    # ✅ Only embed if empty
    if rag_pipeline.energy_collection.count() == 0:
        rag_pipeline.ingest_energy_data(df)
        rag_pipeline.ingest_patterns(df)
    else:
        print("✅ Sample data already embedded. Skipping ingestion.")

    return df


# Try to load data, generate sample if not found
energy_df = load_and_ingest_data()

# ---------------------------
# Shared helper: watsonx chat with RAG context
# ---------------------------
def wx_chat_with_rag(messages: List[Dict[str, str]], rag_context: str = "", **params) -> str:
    """Call watsonx.ai chat with RAG context and return string content."""
    # Inject RAG context into system message
    if rag_context and messages:
        if messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\nRelevant historical data:\n{rag_context}"
        else:
            messages.insert(0, {"role": "system", "content": f"Relevant historical data:\n{rag_context}"})
    
    return wx_chat(messages, **params)

def wx_chat(messages: List[Dict[str, str]], **params) -> str:
    """Call watsonx.ai chat and return string content."""
    default = dict(max_new_tokens=800, temperature=0.2, top_p=1.0, top_k=50)
    default.update(params)
    resp = llm.chat(messages=messages, params=default)
    content = resp["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text.append(part["text"])
            elif isinstance(part, str):
                text.append(part)
        return "".join(text)
    return str(content)

# ---------------------------
# Updated Pydantic schemas for new dataset
# ---------------------------

class TimePoint(BaseModel):
    ts: str                   # ISO datetime string
    total_mw: float
    industrial_mw: Optional[float] = None
    households_mw: Optional[float] = None
    schools_mw: Optional[float] = None
    colleges_mw: Optional[float] = None
    hospitals_mw: Optional[float] = None

class ForecastRequest(BaseModel):
    city: str
    area: Optional[str] = None  # Can forecast for specific area or entire city
    horizon_hours: int = 24
    weather_hint: Optional[str] = None

class Prep3hRequest(BaseModel):
    city: str
    area: Optional[str] = None
    current_load_mw: float
    reserve_margin_pct: float = 15.0
    notes: Optional[str] = None

class LiveSource(BaseModel):
    name: str
    kind: str                 # solar/wind/hydro/rooftop/etc
    current_mw: float
    expected_mw: Optional[float] = None

class MonitorRequest(BaseModel):
    window_minutes: int = 15
    sources: conlist(LiveSource, min_length=1)

class OptimizeRequest(BaseModel):
    city: str
    area: Optional[str] = None
    demand_forecast_mw: conlist(float, min_length=3)  # next few hours
    dispatchable_mw: float
    renewables_now_mw: float
    constraints: Optional[str] = Field(
        default="Respect reserve margin 15%, avoid shedding if possible."
    )

# ---------------------------
# Updated Tool 1: Consumption prediction with RAG
# ---------------------------
@app.tool()
def predict_consumption(req: ForecastRequest) -> Dict[str, Any]:
    """
    Predict electricity consumption for a city/area using RAG pipeline.
    Returns JSON with hourly forecast and a path to a PNG plot.
    """
    # Get historical context from RAG
    query = f"electricity consumption forecast {req.city}"
    if req.area:
        query += f" {req.area}"
    
    historical_results = rag_pipeline.query_similar_data(query, "energy_consumption", n_results=50)
    pattern_results = rag_pipeline.query_similar_data(query, "consumption_patterns", n_results=5)
    
    # Build RAG context
    rag_context = "Historical consumption data:\n"
    for i, doc in enumerate(historical_results['documents'][0][:10]):
        rag_context += f"- {doc[:200]}...\n"
    
    rag_context += "\nConsumption patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:300]}...\n"
    
    # Get recent data for the specific city/area
    recent_data = []
    if req.area:
        city_area_data = energy_df[(energy_df['City'] == req.city) & (energy_df['Area'] == req.area)]
    else:
        city_area_data = energy_df[energy_df['City'] == req.city]
    
    if not city_area_data.empty:
        # Get last 48 hours worth of data
        recent_data = city_area_data.tail(48).to_dict('records')
    
    # Prompt the LLM with RAG context
    sys = (
        f"You are a power-systems forecasting assistant with access to historical data. "
        f"Generate a realistic hourly electricity consumption forecast for {req.city} "
        f"{'area ' + req.area if req.area else ''} for the next {req.horizon_hours} hours. "
        f"Consider consumption patterns for Industrial, Households, Schools, Colleges, and Hospitals. "
        f"Return JSON with 'forecast' array of {req.horizon_hours} MW values and 'breakdown' with "
        f"Industrial, Households, Schools, Colleges, Hospitals for each hour."
    )
    
    user = {
        "city": req.city,
        "area": req.area,
        "horizon_hours": req.horizon_hours,
        "weather_hint": req.weather_hint or "unknown",
        "recent_data": recent_data[-24:] if recent_data else []  # Last 24 hours
    }

    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        rag_context,
        max_new_tokens=1500, temperature=0.1
    )

    # Parse LLM response
    try:
        data = json.loads(raw)
        forecast = data.get("forecast", [])
        breakdown = data.get("breakdown", [])
        
        if not isinstance(forecast, list) or len(forecast) < req.horizon_hours:
            raise ValueError("Invalid forecast format")
        
        forecast = [float(x) for x in forecast[:req.horizon_hours]]
    except Exception as e:
        print(f"LLM forecast parsing failed: {e}, using statistical fallback")
        # Statistical fallback using historical data
        if not recent_data:
            forecast = [150.0] * req.horizon_hours  # Default fallback
        else:
            # Use average patterns from recent data
            avg_total = np.mean([float(d['Total']) for d in recent_data])
            # Apply hourly patterns
            forecast = []
            for h in range(req.horizon_hours):
                hour_of_day = h % 24
                if 6 <= hour_of_day <= 9 or 18 <= hour_of_day <= 22:  # Peak
                    multiplier = 1.2
                elif 23 <= hour_of_day <= 5:  # Low
                    multiplier = 0.7
                else:
                    multiplier = 1.0
                forecast.append(avg_total * multiplier)
        
        breakdown = []

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Total forecast
    hours = list(range(req.horizon_hours))
    ax1.plot(hours, forecast, label="Total Forecast", linewidth=2, color='blue')
    ax1.set_title(f"Electricity Consumption Forecast - {req.city}" + (f" ({req.area})" if req.area else ""), fontsize=14)
    ax1.set_xlabel("Hours Ahead", fontsize=12)
    ax1.set_ylabel("Load (MW)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Breakdown (if available)
    if breakdown and len(breakdown) >= req.horizon_hours:
        sectors = ['Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals']
        bottom = np.zeros(req.horizon_hours)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, sector in enumerate(sectors):
            values = [breakdown[h].get(sector, 0) for h in range(req.horizon_hours)]
            ax2.bar(hours, values, bottom=bottom, label=sector, color=colors[i % len(colors)])
            bottom += values
    else:
        ax2.plot(hours, forecast, label="Total (Breakdown N/A)", linewidth=2, color='orange')
    
    ax2.set_title("Consumption Breakdown by Sector", fontsize=14)
    ax2.set_xlabel("Hours Ahead", fontsize=12)
    ax2.set_ylabel("Load (MW)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    out_path = os.path.join(tempfile.gettempdir(), f"forecast_{uuid.uuid4().hex}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "city": req.city,
        "area": req.area,
        "horizon_hours": req.horizon_hours,
        "forecast_mw": forecast,
        "breakdown": breakdown if breakdown else None,
        "plot_path": out_path,
        "data_points_used": len(recent_data)
    }

# ---------------------------
# Updated remaining tools (keeping similar structure but adapting to new data format)
# ---------------------------

@app.tool()
def prep_energy_3h(req: Prep3hRequest) -> Dict[str, Any]:
    """
    Compute recommended prep capacity for next 3 hours in a city/area with RAG context.
    """
    # Get relevant context from RAG
    query = f"energy preparation planning {req.city}"
    if req.area:
        query += f" {req.area}"
    
    context_results = rag_pipeline.query_similar_data(query, "energy_consumption", n_results=20)
    pattern_results = rag_pipeline.query_similar_data(query, "consumption_patterns", n_results=3)
    
    rag_context = "Relevant consumption data:\n"
    for doc in context_results['documents'][0][:5]:
        rag_context += f"- {doc[:150]}...\n"
    
    rag_context += "\nConsumption patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:200]}...\n"
    
    sys = (
        "You are a grid operations planner with access to historical consumption data. "
        "Based on current load and historical patterns, estimate required prepared capacity "
        "for each of the next 3 hours by sector (Industrial, Households, Schools, Colleges, Hospitals). "
        "Return JSON {\"prep_mw\": [h1,h2,h3], \"sector_breakdown\": [{\"Industrial\": x, \"Households\": y, ...}, ...], \"rationale\": \"...\"} only."
    )
    
    user = {
        "city": req.city,
        "area": req.area,
        "current_load_mw": req.current_load_mw,
        "reserve_margin_pct": req.reserve_margin_pct,
        "notes": req.notes or ""
    }
    
    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        rag_context,
        max_new_tokens=600, temperature=0.2
    )
    
    try:
        data = json.loads(raw)
        prep_mw = data.get("prep_mw", [])
        if not isinstance(prep_mw, list) or len(prep_mw) != 3:
            raise ValueError("Invalid prep_mw format")
        prep_mw = [float(x) for x in prep_mw]
    except Exception as e:
        print(f"LLM prep parsing failed: {e}, using fallback")
        base = req.current_load_mw * (1 + req.reserve_margin_pct/100)
        prep_mw = [base, base * 1.05, base * 1.1]
        data = {
            "prep_mw": prep_mw,
            "sector_breakdown": [],
            "rationale": "Fallback: applied reserve margin with hourly increase based on typical demand patterns."
        }
    
    result = {
        "city": req.city,
        "area": req.area,
        "prep_mw": prep_mw,
        "sector_breakdown": data.get("sector_breakdown", []),
        "rationale": data.get("rationale", "Capacity planning based on current load and historical patterns.")
    }
    return result

# Keep the monitor_generation and optimize_load tools largely the same but add RAG context
@app.tool()
def monitor_generation(req: MonitorRequest) -> Dict[str, Any]:
    """
    Summarize live generation from distributed sources and flag anomalies.
    """
    sys = (
        "You are an energy monitoring assistant. Analyze live distributed generation. "
        "Identify shortfalls vs expected, anomalies, and provide status assessment. "
        "Return JSON {\"summary\":\"...\", \"anomalies\":[{\"name\":\"...\",\"issue\":\"...\",\"suggested_action\":\"...\"}], "
        "\"total_now_mw\": number, \"total_expected_mw\": number, \"status\": \"normal|warning|critical\"} only."
    )
    user = {
        "window_minutes": req.window_minutes, 
        "sources": [s.dict() for s in req.sources]
    }
    raw = wx_chat(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        max_new_tokens=600, temperature=0.2
    )
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"LLM monitor parsing failed: {e}, using fallback")
        total_now = sum(s.current_mw for s in req.sources)
        total_exp = sum(s.expected_mw or s.current_mw for s in req.sources)
        
        anomalies = []
        for s in req.sources:
            if s.expected_mw and s.current_mw < s.expected_mw * 0.8:
                anomalies.append({
                    "name": s.name,
                    "issue": f"Underperforming: {s.current_mw:.1f} MW vs expected {s.expected_mw:.1f} MW",
                    "suggested_action": "Check weather conditions and equipment status"
                })
        
        status = "critical" if len(anomalies) > len(req.sources) * 0.5 else ("warning" if anomalies else "normal")
        
        data = {
            "summary": f"Monitoring {len(req.sources)} sources. Total output: {total_now:.1f} MW of {total_exp:.1f} MW expected.",
            "anomalies": anomalies,
            "total_now_mw": total_now,
            "total_expected_mw": total_exp,
            "status": status
        }
    
    return {
        "summary": str(data.get("summary", "")),
        "anomalies": data.get("anomalies", []),
        "total_now_mw": float(data.get("total_now_mw", 0)),
        "total_expected_mw": float(data.get("total_expected_mw", 0)),
        "status": str(data.get("status", "unknown"))
    }

@app.tool()
def optimize_load(req: OptimizeRequest) -> Dict[str, Any]:
    """
    Recommend actions to balance supply/demand with minimal curtailment using RAG context.
    """
    # Get relevant optimization context
    query = f"load optimization {req.city}"
    if req.area:
        query += f" {req.area}"
    
    context_results = rag_pipeline.query_similar_data(query, "consumption_patterns", n_results=5)
    
    rag_context = "Consumption patterns for optimization:\n"
    for doc in context_results['documents'][0]:
        rag_context += f"- {doc[:200]}...\n"
    
    sys = (
        "You are a grid optimization expert with access to consumption patterns. "
        "Recommend 3-6 concrete actions considering sector-specific load patterns "
        "(Industrial, Households, Schools, Colleges, Hospitals). "
        "Return JSON {\"actions\":[{\"title\":\"...\",\"mw_effect\":number,\"eta_min\":number,\"cost_level\":\"low|medium|high\",\"sector\":\"...\",\"notes\":\"...\"}], "
        "\"expected_net_relief_mw\": number, \"residual_risk\":\"low|medium|high\", \"rationale\":\"...\"} only."
    )
    
    user_data = req.dict()
    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_data)}
        ],
        rag_context,
        max_new_tokens=800, temperature=0.25
    )
    
    try:
        data = json.loads(raw)
        actions = data.get("actions", [])
        for action in actions:
            action["mw_effect"] = float(action.get("mw_effect", 0))
            action["eta_min"] = int(action.get("eta_min", 30))
            action["cost_level"] = str(action.get("cost_level", "medium"))
            action["sector"] = str(action.get("sector", "general"))
    except Exception as e:
        print(f"LLM optimization parsing failed: {e}, using fallback")
        peak_demand = max(req.demand_forecast_mw)
        available_supply = req.dispatchable_mw + req.renewables_now_mw
        shortfall = max(0, peak_demand - available_supply)
        
        actions = []
        if shortfall > 0:
            actions.append({
                "title": "Industrial demand response program",
                "mw_effect": min(shortfall * 0.4, 150),
                "eta_min": 30,
                "cost_level": "medium",
                "sector": "Industrial",
                "notes": "Target large industrial consumers with time-of-use pricing"
            })
            actions.append({
                "title": "Residential peak load management",
                "mw_effect": min(shortfall * 0.3, 100),
                "eta_min": 45,
                "cost_level": "low",
                "sector": "Households",
                "notes": "Smart home automation and AC temperature adjustments"
            })
        else:
            actions.append({
                "title": "Maintain current operations",
                "mw_effect": 0,
                "eta_min": 0,
                "cost_level": "low",
                "sector": "general",
                "notes": "Grid is balanced, continue monitoring"
            })
        
        net_relief = sum(action["mw_effect"] for action in actions)
        risk = "low" if shortfall <= net_relief else ("medium" if shortfall <= net_relief * 1.2 else "high")
        
        data = {
            "actions": actions,
            "expected_net_relief_mw": net_relief,
            "residual_risk": risk,
            "rationale": f"Grid analysis shows {shortfall:.0f} MW shortfall. Sector-specific actions provide {net_relief:.0f} MW relief."
        }
    
    result = {
        "city": req.city,
        "area": req.area,
        "actions": data.get("actions", []),
        "expected_net_relief_mw": float(data.get("expected_net_relief_mw", 0)),
        "residual_risk": str(data.get("residual_risk", "medium")),
        "rationale": str(data.get("rationale", "Load optimization analysis completed."))
    }
    return result

# ---------------------------
# Additional RAG-enabled tools
# ---------------------------

@app.tool()
def analyze_consumption_patterns(city: str, area: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze consumption patterns for a city or specific area using RAG.
    """
    query = f"consumption patterns analysis {city}"
    if area:
        query += f" {area}"
    
    pattern_results = rag_pipeline.query_similar_data(query, "consumption_patterns", n_results=10)
    energy_results = rag_pipeline.query_similar_data(query, "energy_consumption", n_results=50)
    
    # Get actual data for analysis
    if area:
        data = energy_df[(energy_df['City'] == city) & (energy_df['Area'] == area)]
    else:
        data = energy_df[energy_df['City'] == city]
    
    analysis = {}
    if not data.empty:
        # Calculate statistics
        analysis = {
            "peak_consumption": {
                "value": float(data['Total'].max()),
                "time": data.loc[data['Total'].idxmax(), 'Hour'],
                "area": data.loc[data['Total'].idxmax(), 'Area'] if not area else area
            },
            "low_consumption": {
                "value": float(data['Total'].min()),
                "time": data.loc[data['Total'].idxmin(), 'Hour'],
                "area": data.loc[data['Total'].idxmin(), 'Area'] if not area else area
            },
            "average_by_sector": {
                "Industrial": float(data['Industrial'].mean()),
                "Households": float(data['Households'].mean()),
                "Schools": float(data['Schools'].mean()),
                "Colleges": float(data['Colleges'].mean()),
                "Hospitals": float(data['Hospitals'].mean()),
                "Total": float(data['Total'].mean())
            },
            "hourly_patterns": data.groupby('Hour')['Total'].mean().to_dict()
        }
    
    # Build context for LLM analysis
    rag_context = "Historical patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:300]}...\n"
    
    sys = (
        "You are an energy consumption analyst. Analyze the consumption patterns and provide insights. "
        "Focus on peak hours, sector contributions, and recommendations for optimization. "
        "Return JSON with 'insights', 'recommendations', and 'key_findings' arrays."
    )
    
    user = {
        "city": city,
        "area": area,
        "analysis_data": analysis
    }
    
    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        rag_context,
        max_new_tokens=800, temperature=0.3
    )
    
    try:
        llm_analysis = json.loads(raw)
    except:
        llm_analysis = {
            "insights": ["Pattern analysis completed"],
            "recommendations": ["Continue monitoring consumption patterns"],
            "key_findings": ["Data analysis performed successfully"]
        }
    
    return {
        "city": city,
        "area": area,
        "statistical_analysis": analysis,
        "ai_insights": llm_analysis
    }

@app.tool()
def get_similar_consumption_days(city: str, target_date: str, area: Optional[str] = None, n_days: int = 5) -> Dict[str, Any]:
    """
    Find days with similar consumption patterns using RAG similarity search.
    """
    # Get data for the target date
    target_data = energy_df[(energy_df['City'] == city) & (energy_df['Date'] == target_date)]
    if area:
        target_data = target_data[target_data['Area'] == area]
    
    if target_data.empty:
        return {
            "error": f"No data found for {city} on {target_date}"
        }
    
    # Calculate total consumption for target date
    target_total = float(target_data['Total'].sum())
    target_pattern = target_data.groupby('Hour')['Total'].sum().to_dict()
    
    # Create query for similar patterns
    query = f"consumption pattern {city} total {target_total:.0f} MW"
    if area:
        query += f" {area}"
    
    similar_results = rag_pipeline.query_similar_data(query, "energy_consumption", n_results=200)
    
    # Extract dates from similar results and find unique dates
    similar_dates = set()
    for metadata in similar_results['metadatas'][0]:
        date = metadata.get('date')
        if date and date != target_date:
            similar_dates.add(date)
    
    # Get consumption data for similar dates
    similar_days = []
    for date in list(similar_dates)[:n_days]:
        date_data = energy_df[(energy_df['City'] == city) & (energy_df['Date'] == date)]
        if area:
            date_data = date_data[date_data['Area'] == area]
        
        if not date_data.empty:
            similar_days.append({
                "date": date,
                "total_consumption": float(date_data['Total'].sum()),
                "peak_hour": date_data.loc[date_data['Total'].idxmax(), 'Hour'],
                "peak_value": float(date_data['Total'].max()),
                "sector_breakdown": {
                    "Industrial": float(date_data['Industrial'].sum()),
                    "Households": float(date_data['Households'].sum()),
                    "Schools": float(date_data['Schools'].sum()),
                    "Colleges": float(date_data['Colleges'].sum()),
                    "Hospitals": float(date_data['Hospitals'].sum())
                }
            })
    
    return {
        "target_date": target_date,
        "target_total_consumption": target_total,
        "target_pattern": target_pattern,
        "similar_days": similar_days,
        "city": city,
        "area": area
    }

# ---------------------------
# Run MCP server (stdio)
# ---------------------------
if __name__ == "__main__":
    print("🚀 Grid Operations MCP Server with RAG Pipeline")
    print("📊 ChromaDB initialized for vector embeddings")
    print("🔍 Energy data ingested and ready for queries")
    app.run(transport="stdio")