import os, json, uuid, tempfile, re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# LangChain imports
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import logging

from pydantic import BaseModel, Field, conlist

# MCP server
from mcp.server.fastmcp import FastMCP

# watsonx.ai SDK
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pymongo import MongoClient

# Load config
load_dotenv()
API_KEY = os.environ["WATSONX_API_KEY"]
BASE_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]
MODEL_ID = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

creds = Credentials(api_key=API_KEY, url=BASE_URL)
llm = ModelInference(model_id=MODEL_ID, credentials=creds, project_id=PROJECT_ID)

app = FastMCP("grid-ops")
path_to_csv = "sample_data/load_history1.csv"

class RAGPipeline:
    def __init__(self, persist_directory="./faiss_db"):
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector stores
        self.energy_vectorstore = None
        self.patterns_vectorstore = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def ingest_energy_data(self, df: pd.DataFrame):
        """Ingest energy consumption data into FAISS vectorstore"""
        documents = []
        
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
            
            metadata = {
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
            }
            
            documents.append(Document(page_content=doc_text, metadata=metadata))
        
        # Create FAISS vectorstore
        if documents:
            self.energy_vectorstore = FAISS.from_documents(documents, self.embeddings)
            # Save to disk
            os.makedirs(self.persist_directory, exist_ok=True)
            self.energy_vectorstore.save_local(os.path.join(self.persist_directory, "energy"))
        
        self.logger.info(f"Ingested {len(documents)} energy consumption records")

    def ingest_patterns(self, df: pd.DataFrame):
        """Extract and ingest consumption patterns"""
        pattern_docs = []
        
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
            
            metadata = {
                "area": area,
                "pattern_type": "daily",
                "peak_hour": daily_pattern['Total'].idxmax(),
                "peak_mw": float(daily_pattern['Total'].max()),
                "low_hour": daily_pattern['Total'].idxmin(),
                "low_mw": float(daily_pattern['Total'].min()),
                "avg_total": float(daily_pattern['Total'].mean())
            }
            
            pattern_docs.append(Document(page_content=pattern_text, metadata=metadata))
        
        if pattern_docs:
            self.patterns_vectorstore = FAISS.from_documents(pattern_docs, self.embeddings)
            self.patterns_vectorstore.save_local(os.path.join(self.persist_directory, "patterns"))
        
        self.logger.info(f"Ingested {len(pattern_docs)} consumption patterns")

    def load_vectorstores(self):
        """Load existing vectorstores if they exist"""
        try:
            energy_path = os.path.join(self.persist_directory, "energy")
            patterns_path = os.path.join(self.persist_directory, "patterns")
            
            if os.path.exists(energy_path):
                self.energy_vectorstore = FAISS.load_local(energy_path, self.embeddings, allow_dangerous_deserialization=True)
                self.logger.info("Loaded existing energy vectorstore")
            
            if os.path.exists(patterns_path):
                self.patterns_vectorstore = FAISS.load_local(patterns_path, self.embeddings, allow_dangerous_deserialization=True)
                self.logger.info("Loaded existing patterns vectorstore")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing vectorstores: {e}")

    def query_similar_data(self, query: str, collection_type: str = "energy", k: int = 10):
        """Query similar data from vectorstore"""
        try:
            if collection_type == "energy" and self.energy_vectorstore:
                docs = self.energy_vectorstore.similarity_search(query, k=k)
                return {
                    'documents': [[doc.page_content for doc in docs]],
                    'metadatas': [[doc.metadata for doc in docs]]
                }
            elif collection_type == "patterns" and self.patterns_vectorstore:
                docs = self.patterns_vectorstore.similarity_search(query, k=k)
                return {
                    'documents': [[doc.page_content for doc in docs]],
                    'metadatas': [[doc.metadata for doc in docs]]
                }
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
        
        return {'documents': [[]], 'metadatas': [[]]}

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

def load_and_ingest_data(csv_path: str = path_to_csv):
    """Load CSV data and ingest into RAG pipeline if not already embedded"""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using sample data generation.")
        return generate_sample_data()
    
    df = pd.read_csv(csv_path)
    
    # Try to load existing vectorstores first
    rag_pipeline.load_vectorstores()
    
    # Only embed if vectorstore doesn't exist
    if rag_pipeline.energy_vectorstore is None:
        print("⚡ No embeddings found. Ingesting energy data...")
        rag_pipeline.ingest_energy_data(df)
        rag_pipeline.ingest_patterns(df)
    else:
        print("✅ Embeddings already exist. Skipping ingestion.")
    
    return df

def generate_sample_data():
    """Generate sample data matching the new format"""
    cities = ["Bangalore"]
    areas = ["MG Road", "Brigade Road", "Church Street", "Richmond Town", 
             "Shivajinagar", "Vasanth Nagar", "Ulsoor (Halasuru)", 
             "Indiranagar", "Koramangala", "Malleshwaram", "Rajajinagar"]
    
    data = []
    start_date = datetime(2025, 1, 1)
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        for hour in range(24):
            for city in cities:
                for area in areas:
                    base_industrial = np.random.uniform(50, 90)
                    base_households = np.random.uniform(30, 50)
                    base_schools = np.random.uniform(1, 3) if 8 <= hour <= 16 else np.random.uniform(0.5, 1.5)
                    base_colleges = np.random.uniform(2, 5) if 9 <= hour <= 17 else np.random.uniform(0.5, 2)
                    base_hospitals = np.random.uniform(15, 30)
                    
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        multiplier = 1.2
                    elif 23 <= hour <= 5:
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

    rag_pipeline.load_vectorstores()
    
    if rag_pipeline.energy_vectorstore is None:
        rag_pipeline.ingest_energy_data(df)
        rag_pipeline.ingest_patterns(df)
    else:
        print("✅ Sample data already embedded. Skipping ingestion.")

    return df

energy_df = load_and_ingest_data()

def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text using multiple strategies"""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*```', '', text)
    text = text.strip()

    # FIXED: Only remove units when they're clearly not part of JSON values
    # Be more careful about removing MW and other units
    text = re.sub(r'\s+(MW|mw|MWs)\s*(?=[,}\]\s])', '', text)

    # Strategy 1: Look for balanced braces
    brace_count = 0
    start_pos = -1
    for i, char in enumerate(text):
        if char == '{':
            if start_pos == -1:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                candidate = text[start_pos:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

    # Strategy 2: Line-by-line reconstruction
    lines = text.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '{' in line and not in_json:
            in_json = True
            json_lines = [line]
            brace_count = line.count('{') - line.count('}')
        elif in_json:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                candidate = ' '.join(json_lines)
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
                break

    # Strategy 3: Simple first/last brace
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]

    return text

def fix_json_format(json_str: str) -> str:
    """Fix common JSON formatting issues with enhanced error handling"""
    try:
        original = json_str
        json_str = json_str.strip()

        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Quote keys properly
        json_str = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', json_str)

        # Convert single quotes → double quotes
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        # Normalize booleans & nulls
        json_str = re.sub(r':\s*(True|TRUE|true)\b', ': true', json_str)
        json_str = re.sub(r':\s*(False|FALSE|false)\b', ': false', json_str)
        json_str = re.sub(r':\s*(None|NULL|null)\b', ': null', json_str)

        # FIXED: Be very specific about time patterns - only fix actual time formats
        # Don't mess with sector values that might look like time patterns
        json_str = re.sub(r'"(\d{1,2})":\s*"(\d{2})"(?=\s*[,}])', r'"\1:\2"', json_str)

        # Missing commas between objects/arrays
        json_str = re.sub(r'}\s*{', '}, {', json_str)
        json_str = re.sub(r']\s*\[', '], [', json_str)

        # Cleanup whitespace
        json_str = re.sub(r'\s+', ' ', json_str)

        # Validate
        json.loads(json_str)
        return json_str

    except json.JSONDecodeError as e:
        print(f"JSON parsing error at position {e.pos}: {e.msg}")
        return original
    except Exception as e:
        print(f"JSON fix error: {e}")
        return original

def clean_json_response(raw_response: str) -> str:
    """Enhanced JSON cleaning with multiple fallback strategies"""
    try:
        # First, try to extract JSON from the raw response
        json_candidate = extract_json_from_text(raw_response)
        
        # Apply formatting fixes
        fixed_json = fix_json_format(json_candidate)
        
        # Test if it's valid
        json.loads(fixed_json)
        return fixed_json
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed at position {e.pos}: {e.msg}")
        print(f"Problematic area: {fixed_json[max(0, e.pos-20):e.pos+20]}")
        
        # Try more aggressive cleaning
        try:
            # Remove any non-JSON content before and after
            lines = raw_response.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                    brace_count = line.count('{') - line.count('}')
                    json_lines.append(line)
                elif in_json:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        break
            
            if json_lines:
                candidate = '\n'.join(json_lines)
                candidate = fix_json_format(candidate)
                json.loads(candidate)  # Test validity
                return candidate
                
        except:
            pass
        
        # Final fallback - return original
        return raw_response
    
    except Exception as e:
        print(f"Unexpected error in JSON cleaning: {e}")
        return raw_response

def parse_prep_response_fallback(raw_response: str, req) -> Dict[str, Any]:
    """Fallback parser for prep energy responses when JSON fails"""
    try:
        # Try to extract numbers from the response using regex
        numbers = re.findall(r'\b\d+\.?\d*\b', raw_response)
        numbers = [float(n) for n in numbers if float(n) > 0 and float(n) < 10000]  # Reasonable MW range
        
        # If we have at least 3 numbers, use first 3 as prep_mw
        if len(numbers) >= 3:
            prep_mw = numbers[:3]
        else:
            # Generate based on current load and reserve margin
            base = req.current_load_mw * (1 + req.reserve_margin_pct/100)
            prep_mw = [base, base * 1.05, base * 1.1]
        
        # Generate sector breakdown
        sector_breakdown = []
        for total in prep_mw:
            sector_breakdown.append({
                'Industrial': total * 0.4,
                'Households': total * 0.35,
                'Schools': total * 0.05,
                'Colleges': total * 0.08,
                'Hospitals': total * 0.12
            })
        
        # Try to extract rationale from text
        rationale_match = re.search(r'(?:rationale|reason|explanation)[\s:]+(.+?)(?:\.|$)', raw_response, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()[:200]  # Limit length
        else:
            rationale = "Capacity planning based on current load and historical patterns with standard sector distribution."
        
        return {
            "prep_mw": prep_mw,
            "sector_breakdown": sector_breakdown,
            "rationale": rationale
        }
        
    except Exception as e:
        print(f"Fallback parsing also failed: {e}")
        # Ultimate fallback
        base = req.current_load_mw * (1 + req.reserve_margin_pct/100)
        prep_mw = [base, base * 1.05, base * 1.1]
        
        return {
            "prep_mw": prep_mw,
            "sector_breakdown": [
                {'Industrial': p * 0.4, 'Households': p * 0.35, 'Schools': p * 0.05, 'Colleges': p * 0.08, 'Hospitals': p * 0.12}
                for p in prep_mw
            ],
            "rationale": "Fallback: Applied standard reserve margin with hourly growth pattern."
        }

def safe_get_value(data: dict, key: str, default_value=None, value_type=str):
    """Safely get a value from dict with type conversion and default"""
    try:
        value = data.get(key, default_value)
        if value is None or value == "undefined" or value == "":
            return default_value
        
        if value_type == float:
            return float(value)
        elif value_type == int:
            return int(value)
        elif value_type == str:
            return str(value)
        else:
            return value
    except (ValueError, TypeError):
        return default_value

def wx_chat_with_rag(messages: List[Dict[str, str]], rag_context: str = "", **params) -> str:
    """Call watsonx.ai chat with RAG context and return string content."""
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

class TimePoint(BaseModel):
    ts: str
    total_mw: float
    industrial_mw: Optional[float] = None
    households_mw: Optional[float] = None
    schools_mw: Optional[float] = None
    colleges_mw: Optional[float] = None
    hospitals_mw: Optional[float] = None

class ForecastRequest(BaseModel):
    city: str
    area: Optional[str] = None
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
    kind: str
    current_mw: float
    expected_mw: Optional[float] = None

class MonitorRequest(BaseModel):
    window_minutes: int = 15
    sources: conlist(LiveSource, min_length=1)

class OptimizeRequest(BaseModel):
    city: str
    area: Optional[str] = None
    demand_forecast_mw: conlist(float, min_length=3)
    dispatchable_mw: float
    renewables_now_mw: float
    constraints: Optional[str] = Field(
        default="Respect reserve margin 15%, avoid shedding if possible."
    )

@app.tool()
def predict_consumption(req: ForecastRequest) -> Dict[str, Any]:
    """
    Predict electricity consumption for a city/area using RAG pipeline.
    Returns JSON with hourly forecast and a path to a PNG plot.
    """
    query = f"electricity consumption forecast {req.city}"
    if req.area:
        query += f" {req.area}"
    
    historical_results = rag_pipeline.query_similar_data(query, "energy", k=50)
    pattern_results = rag_pipeline.query_similar_data(query, "patterns", k=5)
    
    rag_context = "Historical consumption data:\n"
    for i, doc in enumerate(historical_results['documents'][0][:10]):
        rag_context += f"- {doc[:200]}...\n"
    
    rag_context += "\nConsumption patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:300]}...\n"
    
    recent_data = []
    if req.area:
        city_area_data = energy_df[(energy_df['City'] == req.city) & (energy_df['Area'] == req.area)]
    else:
        city_area_data = energy_df[energy_df['City'] == req.city]
    
    if not city_area_data.empty:
        recent_data = city_area_data.tail(48).to_dict('records')
    
    sys = f"""
You are a power-systems forecasting assistant. Generate a realistic hourly electricity
consumption forecast for {req.city} {'area ' + req.area if req.area else ''} for the next
{req.horizon_hours} hours.

CRITICAL: Return ONLY a valid JSON object with exactly this structure (no additional text):
{{
  "forecast": [{", ".join(["NUMBER"] * req.horizon_hours)}],
  "breakdown": [{", ".join(['{{"Industrial": NUMBER, "Households": NUMBER, "Schools": NUMBER, "Colleges": NUMBER, "Hospitals": NUMBER}}'] * req.horizon_hours)}]
}}

Each NUMBER must be a valid decimal number (like 150.5). Do not use words like 'undefined'.
"""

    
    user = {
        "city": req.city,
        "area": req.area,
        "horizon_hours": req.horizon_hours,
        "weather_hint": req.weather_hint or "normal conditions",
        "recent_data": recent_data[-24:] if recent_data else []
    }

    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Generate forecast for: {json.dumps(user)}"}
        ],
        rag_context,
        max_new_tokens=2000, temperature=0.05
    )

    try:
        clean_json = clean_json_response(raw)
        print(f"DEBUG: Cleaned JSON: {clean_json[:300]}...")
        
        data = json.loads(clean_json)
        forecast = data.get("forecast", [])
        breakdown = data.get("breakdown", [])
        
        # Validate and clean forecast
        if not isinstance(forecast, list) or len(forecast) < req.horizon_hours:
            raise ValueError(f"Invalid forecast format: got {len(forecast) if isinstance(forecast, list) else 'not a list'} items, expected {req.horizon_hours}")
        
        # Convert to float and validate
        clean_forecast = []
        for i, val in enumerate(forecast[:req.horizon_hours]):
            try:
                if isinstance(val, str) and ("undefined" in val.lower() or val.strip() == ""):
                    raise ValueError("Invalid forecast value")
                clean_val = float(val)
                clean_forecast.append(clean_val)
            except (ValueError, TypeError):
                # Use fallback value based on historical data or reasonable estimate
                if recent_data:
                    hour_of_day = i % 24
                    similar_hour_data = [d for d in recent_data if int(d['Hour'].split(':')[0]) == hour_of_day]
                    if similar_hour_data:
                        fallback_val = float(similar_hour_data[-1]['Total'])
                    else:
                        fallback_val = 120.0 + (i % 24) * 2  # Basic pattern
                else:
                    fallback_val = 120.0 + (i % 24) * 2
                clean_forecast.append(fallback_val)
        
        forecast = clean_forecast
        
        # Validate and clean breakdown
        clean_breakdown = []
        if breakdown and isinstance(breakdown, list):
            for i, hour_breakdown in enumerate(breakdown[:req.horizon_hours]):
                if isinstance(hour_breakdown, dict):
                    clean_hour = {}
                    for sector in ['Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals']:
                        try:
                            val = hour_breakdown.get(sector, 0)
                            if isinstance(val, str) and ("undefined" in val.lower() or val.strip() == ""):
                                val = 0
                            clean_hour[sector] = float(val)
                        except (ValueError, TypeError):
                            # Fallback based on typical sector distribution
                            total = forecast[i] if i < len(forecast) else 120
                            sector_ratios = {
                                'Industrial': 0.4, 'Households': 0.35, 'Schools': 0.05,
                                'Colleges': 0.08, 'Hospitals': 0.12
                            }
                            clean_hour[sector] = total * sector_ratios.get(sector, 0.2)
                    clean_breakdown.append(clean_hour)
                else:
                    # Generate fallback breakdown
                    total = forecast[i] if i < len(forecast) else 120
                    clean_breakdown.append({
                        'Industrial': total * 0.4,
                        'Households': total * 0.35,
                        'Schools': total * 0.05,
                        'Colleges': total * 0.08,
                        'Hospitals': total * 0.12
                    })
        
        breakdown = clean_breakdown if clean_breakdown else []
            
    except Exception as e:
        print(f"LLM forecast parsing failed: {e}")
        print(f"Raw LLM response: {raw[:500]}...")
        
        # Enhanced statistical fallback
        if not recent_data:
            base_load = 150.0
            forecast = []
            for h in range(req.horizon_hours):
                hour_of_day = h % 24
                if 6 <= hour_of_day <= 9 or 18 <= hour_of_day <= 22:
                    multiplier = 1.3
                elif 23 <= hour_of_day <= 5:
                    multiplier = 0.6
                else:
                    multiplier = 1.0
                forecast.append(base_load * multiplier + np.random.uniform(-10, 10))
        else:
            # Use recent data patterns for better fallback
            recent_totals = [float(d['Total']) for d in recent_data]
            avg_total = np.mean(recent_totals)
            std_total = np.std(recent_totals)
            
            # Get hourly patterns from recent data
            hour_patterns = {}
            for d in recent_data:
                hour = int(d['Hour'].split(':')[0])
                if hour not in hour_patterns:
                    hour_patterns[hour] = []
                hour_patterns[hour].append(float(d['Total']))
            
            forecast = []
            for h in range(req.horizon_hours):
                hour_of_day = h % 24
                if hour_of_day in hour_patterns:
                    hour_avg = np.mean(hour_patterns[hour_of_day])
                else:
                    hour_avg = avg_total
                
                # Add some realistic variation
                variation = np.random.normal(0, std_total * 0.1)
                forecast.append(max(10, hour_avg + variation))
        
        # Generate breakdown from forecast
        breakdown = []
        for total in forecast:
            breakdown.append({
                'Industrial': total * 0.4,
                'Households': total * 0.35,
                'Schools': total * 0.05,
                'Colleges': total * 0.08,
                'Hospitals': total * 0.12
            })

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    hours = list(range(req.horizon_hours))
    ax1.plot(hours, forecast, label="Total Forecast", linewidth=2, color='blue')
    ax1.set_title(f"Electricity Consumption Forecast - {req.city}" + (f" ({req.area})" if req.area else ""), fontsize=14)
    ax1.set_xlabel("Hours Ahead", fontsize=12)
    ax1.set_ylabel("Load (MW)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
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
        "breakdown": breakdown,
        "plot_path": out_path,
        "data_points_used": len(recent_data)
    }

@app.tool()
def prep_energy_3h(req: Prep3hRequest) -> Dict[str, Any]:
    """
    Compute recommended prep capacity for next 3 hours in a city/area with RAG context.
    """
    query = f"energy preparation planning {req.city}"
    if req.area:
        query += f" {req.area}"
    
    context_results = rag_pipeline.query_similar_data(query, "energy", k=20)
    pattern_results = rag_pipeline.query_similar_data(query, "patterns", k=3)
    
    # Calculate realistic base values from RAG context
    historical_totals = []
    sector_averages = {'Industrial': 0, 'Households': 0, 'Schools': 0, 'Colleges': 0, 'Hospitals': 0}
    sector_counts = {'Industrial': 0, 'Households': 0, 'Schools': 0, 'Colleges': 0, 'Hospitals': 0}
    
    for metadata in context_results['metadatas'][0][:15]:  # Use more samples for better averages
        total_mw = metadata.get('total_mw', 0)
        if total_mw > 0:
            historical_totals.append(total_mw)
        
        # Accumulate sector values
        for sector in sector_averages:
            key = sector.lower()
            if key in metadata and metadata[key] > 0:
                sector_averages[sector] += metadata[key]
                sector_counts[sector] += 1
    
    # Calculate averages
    if historical_totals:
        avg_historical_total = sum(historical_totals) / len(historical_totals)
        for sector in sector_averages:
            if sector_counts[sector] > 0:
                sector_averages[sector] = sector_averages[sector] / sector_counts[sector]
            else:
                # Fallback proportions if no data
                sector_averages[sector] = avg_historical_total * {
                    'Industrial': 0.45, 'Households': 0.32, 'Schools': 0.06, 
                    'Colleges': 0.09, 'Hospitals': 0.08
                }[sector]
    else:
        avg_historical_total = req.current_load_mw
        # Use fallback proportions
        for sector in sector_averages:
            sector_averages[sector] = req.current_load_mw * {
                'Industrial': 0.45, 'Households': 0.32, 'Schools': 0.06,
                'Colleges': 0.09, 'Hospitals': 0.08
            }[sector]
    
    rag_context = "Relevant consumption data:\n"
    for doc in context_results['documents'][0][:5]:
        rag_context += f"- {doc[:150]}...\n"
    
    rag_context += "\nConsumption patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:200]}...\n"
    
    sys = f"""You are a grid operations planner. Based on current load of {req.current_load_mw} MW and historical patterns showing average total of {avg_historical_total:.1f} MW, estimate required prepared capacity for each of the next 3 hours.

Historical sector averages for this location:
- Industrial: {sector_averages['Industrial']:.1f} MW
- Households: {sector_averages['Households']:.1f} MW  
- Schools: {sector_averages['Schools']:.1f} MW
- Colleges: {sector_averages['Colleges']:.1f} MW
- Hospitals: {sector_averages['Hospitals']:.1f} MW

Apply {req.reserve_margin_pct}% reserve margin and account for typical hourly growth patterns.

CRITICAL: Return ONLY a valid JSON object with exactly this structure (no additional text):
{{
  "prep_mw": [HOUR1_TOTAL, HOUR2_TOTAL, HOUR3_TOTAL],
  "sector_breakdown": [
    {{"Industrial": HOUR1_IND, "Households": HOUR1_HH, "Schools": HOUR1_SCH, "Colleges": HOUR1_COL, "Hospitals": HOUR1_HOSP}},
    {{"Industrial": HOUR2_IND, "Households": HOUR2_HH, "Schools": HOUR2_SCH, "Colleges": HOUR2_COL, "Hospitals": HOUR2_HOSP}},
    {{"Industrial": HOUR3_IND, "Households": HOUR3_HH, "Schools": HOUR3_SCH, "Colleges": HOUR3_COL, "Hospitals": HOUR3_HOSP}}
  ],
  "rationale": "TEXT_EXPLANATION"
}}

Each number must be a realistic decimal value. Use the historical averages as a baseline and adjust for reserve margin and hourly patterns."""
    
    user = {
        "city": req.city,
        "area": req.area,
        "current_load_mw": req.current_load_mw,
        "reserve_margin_pct": req.reserve_margin_pct,
        "notes": req.notes or "",
        "historical_context": {
            "avg_total": avg_historical_total,
            "sector_averages": sector_averages
        }
    }
    
    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        rag_context,
        max_new_tokens=1000, temperature=0.1  # Lower temperature for more consistent results
    )
    
    try:
        clean_json = clean_json_response(raw)
        print(f"DEBUG Prep: Cleaned JSON: {clean_json[:300]}...")
        data = json.loads(clean_json)
        
        # Safely extract and validate prep_mw
        prep_mw = data.get("prep_mw", [])
        if not isinstance(prep_mw, list) or len(prep_mw) != 3:
            raise ValueError("Invalid prep_mw format")
        
        clean_prep_mw = []
        for i, val in enumerate(prep_mw):
            try:
                if isinstance(val, str) and ("undefined" in val.lower() or val.strip() == ""):
                    raise ValueError("Invalid prep value")
                clean_val = float(val)
                if clean_val < 10 or clean_val > 2000:  # Sanity check
                    raise ValueError("Unrealistic prep value")
                clean_prep_mw.append(clean_val)
            except (ValueError, TypeError):
                # Fallback calculation based on historical data
                base = max(req.current_load_mw, avg_historical_total) * (1 + req.reserve_margin_pct/100)
                fallback_val = base * (1 + i * 0.03)  # 3% growth per hour
                clean_prep_mw.append(fallback_val)
        
        prep_mw = clean_prep_mw
        
        # Safely extract and validate sector breakdown
        sector_breakdown = data.get("sector_breakdown", [])
        clean_sector_breakdown = []
        
        if isinstance(sector_breakdown, list) and len(sector_breakdown) >= 3:
            for i, hour_data in enumerate(sector_breakdown[:3]):
                if isinstance(hour_data, dict):
                    clean_hour_data = {}
                    hour_total = prep_mw[i] if i < len(prep_mw) else req.current_load_mw
                    
                    for sector in ['Industrial', 'Households', 'Schools', 'Colleges', 'Hospitals']:
                        try:
                            val = hour_data.get(sector, 0)
                            if isinstance(val, str) and ("undefined" in val.lower() or val.strip() == ""):
                                val = 0
                            clean_val = float(val)
                            
                            # Sanity check - ensure values are reasonable
                            max_sector_val = hour_total * 0.8  # No sector should be more than 80% of total
                            if clean_val < 0 or clean_val > max_sector_val:
                                raise ValueError("Unrealistic sector value")
                            
                            clean_hour_data[sector] = round(clean_val, 2)
                        except (ValueError, TypeError):
                            # Use historical average with reserve margin
                            historical_val = sector_averages[sector] * (1 + req.reserve_margin_pct/100)
                            hourly_multiplier = 1 + (i * 0.03)  # 3% increase per hour
                            clean_hour_data[sector] = round(historical_val * hourly_multiplier, 2)
                    
                    # Ensure breakdown sums reasonably close to total
                    breakdown_sum = sum(clean_hour_data.values())
                    if abs(breakdown_sum - hour_total) > hour_total * 0.1:  # More than 10% difference
                        # Proportionally adjust to match total
                        adjustment_factor = hour_total / breakdown_sum
                        for sector in clean_hour_data:
                            clean_hour_data[sector] = round(clean_hour_data[sector] * adjustment_factor, 2)
                    
                    clean_sector_breakdown.append(clean_hour_data)
                else:
                    # Generate fallback breakdown for this hour using historical data
                    hour_total = prep_mw[i] if i < len(prep_mw) else req.current_load_mw
                    hourly_multiplier = 1 + (i * 0.03)
                    
                    clean_sector_breakdown.append({
                        'Industrial': round(sector_averages['Industrial'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                        'Households': round(sector_averages['Households'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                        'Schools': round(sector_averages['Schools'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                        'Colleges': round(sector_averages['Colleges'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                        'Hospitals': round(sector_averages['Hospitals'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2)
                    })
        else:
            # Generate complete fallback breakdown using historical averages
            for i, total in enumerate(prep_mw):
                hourly_multiplier = 1 + (i * 0.03)
                clean_sector_breakdown.append({
                    'Industrial': round(sector_averages['Industrial'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                    'Households': round(sector_averages['Households'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                    'Schools': round(sector_averages['Schools'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                    'Colleges': round(sector_averages['Colleges'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2),
                    'Hospitals': round(sector_averages['Hospitals'] * (1 + req.reserve_margin_pct/100) * hourly_multiplier, 2)
                })
        
        rationale = safe_get_value(data, "rationale", 
            f"Capacity planning based on current load of {req.current_load_mw} MW, historical average of {avg_historical_total:.1f} MW, and {req.reserve_margin_pct}% reserve margin.", str)
        
        final_data = {
            "prep_mw": prep_mw,
            "sector_breakdown": clean_sector_breakdown,
            "rationale": rationale
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed for prep energy: {e}")
        print(f"Raw response: {raw[:300]}...")
        print(f"Attempting enhanced fallback parsing...")
        
        # Enhanced fallback using historical data
        base_multiplier = (1 + req.reserve_margin_pct/100)
        
        prep_mw = []
        sector_breakdown = []
        
        for hour in range(3):
            # Progressive hourly increase
            hour_multiplier = 1.0 + (hour * 0.03)  # 3% increase per hour
            
            # Use historical averages with reserve margin
            industrial = sector_averages['Industrial'] * base_multiplier * hour_multiplier
            households = sector_averages['Households'] * base_multiplier * hour_multiplier
            schools = sector_averages['Schools'] * base_multiplier * hour_multiplier
            colleges = sector_averages['Colleges'] * base_multiplier * hour_multiplier
            hospitals = sector_averages['Hospitals'] * base_multiplier * hour_multiplier
            
            total = industrial + households + schools + colleges + hospitals
            prep_mw.append(round(total, 2))
            
            sector_breakdown.append({
                'Industrial': round(industrial, 2),
                'Households': round(households, 2),
                'Schools': round(schools, 2),
                'Colleges': round(colleges, 2),
                'Hospitals': round(hospitals, 2)
            })
        
        final_data = {
            "prep_mw": prep_mw,
            "sector_breakdown": sector_breakdown,
            "rationale": f"Enhanced fallback planning using historical sector averages: Industrial {sector_averages['Industrial']:.1f} MW, Households {sector_averages['Households']:.1f} MW, with {req.reserve_margin_pct}% reserve margin and progressive hourly increases."
        }
        
    except Exception as e:
        print(f"LLM prep parsing failed: {e}, using ultimate fallback")
        print(f"Raw response: {raw[:300]}...")
        
        # Ultimate fallback with realistic distributions
        base_multiplier = (1 + req.reserve_margin_pct/100)
        base_total = max(req.current_load_mw, avg_historical_total) * base_multiplier
        
        prep_mw = []
        sector_breakdown = []
        
        for hour in range(3):
            hour_multiplier = 1.0 + (hour * 0.05)  # 5% increase per hour
            total = base_total * hour_multiplier
            prep_mw.append(round(total, 2))
            
            # Use historical averages if available, otherwise use realistic proportions
            if sum(sector_averages.values()) > 0:
                sector_breakdown.append({
                    'Industrial': round(sector_averages['Industrial'] * base_multiplier * hour_multiplier, 2),
                    'Households': round(sector_averages['Households'] * base_multiplier * hour_multiplier, 2),
                    'Schools': round(sector_averages['Schools'] * base_multiplier * hour_multiplier, 2),
                    'Colleges': round(sector_averages['Colleges'] * base_multiplier * hour_multiplier, 2),
                    'Hospitals': round(sector_averages['Hospitals'] * base_multiplier * hour_multiplier, 2)
                })
            else:
                sector_breakdown.append({
                    'Industrial': round(total * 0.45, 2),    # 45% - realistic for urban areas
                    'Households': round(total * 0.32, 2),   # 32% - households  
                    'Schools': round(total * 0.06, 2),      # 6% - schools
                    'Colleges': round(total * 0.09, 2),     # 9% - colleges
                    'Hospitals': round(total * 0.08, 2)     # 8% - hospitals  
                })
        
        final_data = {
            "prep_mw": prep_mw,
            "sector_breakdown": sector_breakdown,
            "rationale": f"Ultimate fallback: Applied {req.reserve_margin_pct}% reserve margin to current load {req.current_load_mw} MW with realistic sector distribution based on urban consumption patterns."
        }
    
    return {
        "city": req.city,
        "area": req.area,
        "prep_mw": final_data["prep_mw"],
        "sector_breakdown": final_data["sector_breakdown"],
        "rationale": final_data["rationale"],
        "data_confidence": "high" if sum(sector_averages.values()) > 0 else "medium"
    }

@app.tool()
def monitor_generation(req: MonitorRequest) -> Dict[str, Any]:
    """
    Summarize live generation from distributed sources and flag anomalies.
    """
    sys = (
        "You are an energy monitoring assistant. Analyze live distributed generation. "
        "Identify shortfalls vs expected, anomalies, and provide status assessment.\n\n"
        "CRITICAL: Return ONLY valid JSON with exactly this structure (no additional text):\n"
        "{\n"
        '  "summary": "TEXT_SUMMARY",\n'
        '  "anomalies": [{"name": "SOURCE_NAME", "issue": "ISSUE_DESCRIPTION", "suggested_action": "ACTION_RECOMMENDATION"}],\n'
        '  "total_now_mw": NUMBER,\n'
        '  "total_expected_mw": NUMBER,\n'
        '  "status": "normal|warning|critical"\n'
        "}\n\n"
        "Each NUMBER must be a valid decimal number. Do not use words like 'undefined'."
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
        max_new_tokens=800, temperature=0.2
    )
    
    try:
        clean_json = clean_json_response(raw)
        data = json.loads(clean_json)
        
        # Validate and clean the response
        summary = safe_get_value(data, "summary", "Generation monitoring completed.", str)
        anomalies = data.get("anomalies", [])
        
        # Clean anomalies array
        clean_anomalies = []
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    clean_anomaly = {
                        "name": safe_get_value(anomaly, "name", "Unknown source", str),
                        "issue": safe_get_value(anomaly, "issue", "No specific issue identified", str),
                        "suggested_action": safe_get_value(anomaly, "suggested_action", "Monitor closely", str)
                    }
                    clean_anomalies.append(clean_anomaly)
        
        total_now_mw = safe_get_value(data, "total_now_mw", 
            sum(s.current_mw for s in req.sources), float)
        total_expected_mw = safe_get_value(data, "total_expected_mw", 
            sum(s.expected_mw or s.current_mw for s in req.sources), float)
        
        status = safe_get_value(data, "status", "normal", str)
        if status not in ["normal", "warning", "critical"]:
            status = "normal"
        
        final_data = {
            "summary": summary,
            "anomalies": clean_anomalies,
            "total_now_mw": total_now_mw,
            "total_expected_mw": total_expected_mw,
            "status": status
        }
        
    except Exception as e:
        print(f"LLM monitor parsing failed: {e}, using fallback")
        print(f"Raw response: {raw[:300]}...")
        
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
        
        final_data = {
            "summary": f"Monitoring {len(req.sources)} sources. Total output: {total_now:.1f} MW of {total_exp:.1f} MW expected.",
            "anomalies": anomalies,
            "total_now_mw": total_now,
            "total_expected_mw": total_exp,
            "status": status
        }
    
    return final_data

@app.tool()
def optimize_load(req: OptimizeRequest) -> Dict[str, Any]:
    """
    Recommend actions to balance supply/demand with minimal curtailment using RAG context.
    """
    query = f"load optimization {req.city}"
    if req.area:
        query += f" {req.area}"
    
    context_results = rag_pipeline.query_similar_data(query, "patterns", k=5)
    
    rag_context = "Consumption patterns for optimization:\n"
    for doc in context_results['documents'][0]:
        rag_context += f"- {doc[:200]}...\n"
    
    sys = (
        "You are a grid optimization expert. Recommend 3-6 concrete actions considering "
        "sector-specific load patterns.\n\n"
        "CRITICAL: Return ONLY valid JSON with exactly this structure (no additional text):\n"
        "{\n"
        '  "actions": [{"title": "ACTION_TITLE", "mw_effect": NUMBER, "eta_min": NUMBER, "cost_level": "low|medium|high", "sector": "SECTOR_NAME", "notes": "DESCRIPTION"}],\n'
        '  "expected_net_relief_mw": NUMBER,\n'
        '  "residual_risk": "low|medium|high",\n'
        '  "rationale": "TEXT_EXPLANATION"\n'
        "}\n\n"
        "Each NUMBER must be a valid decimal number. Include 3-6 actions in the actions array."
    )
    
    user_data = req.dict()
    raw = wx_chat_with_rag(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_data)}
        ],
        rag_context,
        max_new_tokens=1000, temperature=0.25
    )
    
    try:
        clean_json = clean_json_response(raw)
        data = json.loads(clean_json)
        
        # Validate and clean actions
        actions = data.get("actions", [])
        clean_actions = []
        
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict):
                    clean_action = {
                        "title": safe_get_value(action, "title", "Load management action", str),
                        "mw_effect": safe_get_value(action, "mw_effect", 0, float),
                        "eta_min": safe_get_value(action, "eta_min", 30, int),
                        "cost_level": safe_get_value(action, "cost_level", "medium", str),
                        "sector": safe_get_value(action, "sector", "general", str),
                        "notes": safe_get_value(action, "notes", "Standard load management", str)
                    }
                    # Validate cost_level
                    if clean_action["cost_level"] not in ["low", "medium", "high"]:
                        clean_action["cost_level"] = "medium"
                    clean_actions.append(clean_action)
        
        # If no valid actions, generate fallback
        if not clean_actions:
            peak_demand = max(req.demand_forecast_mw)
            available_supply = req.dispatchable_mw + req.renewables_now_mw
            shortfall = max(0, peak_demand - available_supply)
            
            if shortfall > 0:
                clean_actions = [
                    {
                        "title": "Industrial demand response program",
                        "mw_effect": min(shortfall * 0.4, 150),
                        "eta_min": 30,
                        "cost_level": "medium",
                        "sector": "Industrial",
                        "notes": "Target large industrial consumers for load reduction"
                    },
                    {
                        "title": "Residential peak load management",
                        "mw_effect": min(shortfall * 0.3, 100),
                        "eta_min": 45,
                        "cost_level": "low",
                        "sector": "Households",
                        "notes": "Smart home automation and AC cycling"
                    }
                ]
            else:
                clean_actions = [
                    {
                        "title": "Maintain current operations",
                        "mw_effect": 0,
                        "eta_min": 0,
                        "cost_level": "low",
                        "sector": "general",
                        "notes": "Grid is currently balanced"
                    }
                ]
        
        net_relief = safe_get_value(data, "expected_net_relief_mw", 
            sum(action["mw_effect"] for action in clean_actions), float)
        
        residual_risk = safe_get_value(data, "residual_risk", "medium", str)
        if residual_risk not in ["low", "medium", "high"]:
            residual_risk = "medium"
        
        rationale = safe_get_value(data, "rationale", 
            "Load optimization analysis completed based on current grid conditions.", str)
        
        final_data = {
            "actions": clean_actions,
            "expected_net_relief_mw": net_relief,
            "residual_risk": residual_risk,
            "rationale": rationale
        }
        
    except Exception as e:
        print(f"LLM optimization parsing failed: {e}, using fallback")
        print(f"Raw response: {raw[:300]}...")
        
        peak_demand = max(req.demand_forecast_mw)
        available_supply = req.dispatchable_mw + req.renewables_now_mw
        shortfall = max(0, peak_demand - available_supply)
        
        actions = []
        if shortfall > 0:
            actions = [
                {
                    "title": "Industrial demand response program",
                    "mw_effect": min(shortfall * 0.4, 150),
                    "eta_min": 30,
                    "cost_level": "medium",
                    "sector": "Industrial",
                    "notes": "Target large industrial consumers for demand reduction"
                },
                {
                    "title": "Residential peak load management",
                    "mw_effect": min(shortfall * 0.3, 100),
                    "eta_min": 45,
                    "cost_level": "low",
                    "sector": "Households",
                    "notes": "Smart home automation and appliance cycling"
                },
                {
                    "title": "Commercial building optimization",
                    "mw_effect": min(shortfall * 0.2, 75),
                    "eta_min": 20,
                    "cost_level": "low",
                    "sector": "Commercial",
                    "notes": "HVAC and lighting adjustments in office buildings"
                }
            ]
        else:
            actions = [
                {
                    "title": "Maintain current operations",
                    "mw_effect": 0,
                    "eta_min": 0,
                    "cost_level": "low",
                    "sector": "general",
                    "notes": "Grid is balanced, no immediate action required"
                }
            ]
        
        net_relief = sum(action["mw_effect"] for action in actions)
        risk = "low" if shortfall <= net_relief else ("medium" if shortfall <= net_relief * 1.2 else "high")
        
        final_data = {
            "actions": actions,
            "expected_net_relief_mw": net_relief,
            "residual_risk": risk,
            "rationale": f"Grid analysis shows {shortfall:.0f} MW shortfall. Recommended actions provide {net_relief:.0f} MW relief."
        }
    
    return {
        "city": req.city,
        "area": req.area,
        "actions": final_data["actions"],
        "expected_net_relief_mw": final_data["expected_net_relief_mw"],
        "residual_risk": final_data["residual_risk"],
        "rationale": final_data["rationale"]
    }

@app.tool()
def analyze_consumption_patterns(city: str, area: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze consumption patterns for a city or specific area using RAG.
    """
    query = f"consumption patterns analysis {city}"
    if area:
        query += f" {area}"
    
    pattern_results = rag_pipeline.query_similar_data(query, "patterns", k=10)
    energy_results = rag_pipeline.query_similar_data(query, "energy", k=50)
    
    if area:
        data = energy_df[(energy_df['City'] == city) & (energy_df['Area'] == area)]
    else:
        data = energy_df[energy_df['City'] == city]
    
    analysis = {}
    if not data.empty:
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
    
    rag_context = "Historical patterns:\n"
    for doc in pattern_results['documents'][0]:
        rag_context += f"- {doc[:300]}...\n"
    
    sys = (
        "You are an energy consumption analyst. Analyze the consumption patterns and provide insights.\n\n"
        "CRITICAL: Return ONLY valid JSON with exactly this structure (no additional text):\n"
        "{\n"
        '  "insights": ["INSIGHT1", "INSIGHT2", "INSIGHT3"],\n'
        '  "recommendations": ["REC1", "REC2", "REC3"],\n'
        '  "key_findings": ["FINDING1", "FINDING2", "FINDING3"]\n'
        "}\n\n"
        "Each array should contain 2-5 string elements."
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
        clean_json = clean_json_response(raw)
        llm_analysis = json.loads(clean_json)
        
        # Validate arrays
        for key in ["insights", "recommendations", "key_findings"]:
            if key not in llm_analysis or not isinstance(llm_analysis[key], list):
                llm_analysis[key] = [f"{key.replace('_', ' ').title()} analysis completed"]
            # Clean any undefined values
            llm_analysis[key] = [
                item for item in llm_analysis[key] 
                if isinstance(item, str) and item.strip() and "undefined" not in item.lower()
            ]
            # Ensure at least one item
            if not llm_analysis[key]:
                llm_analysis[key] = [f"{key.replace('_', ' ').title()} analysis completed"]
                
    except Exception as e:
        print(f"LLM analysis parsing failed: {e}, using fallback")
        llm_analysis = {
            "insights": ["Pattern analysis completed successfully", "Historical data shows regular consumption cycles"],
            "recommendations": ["Continue monitoring consumption patterns", "Implement demand response programs during peak hours"],
            "key_findings": ["Data analysis performed successfully", "Consumption patterns identified for optimization"]
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
    target_data = energy_df[(energy_df['City'] == city) & (energy_df['Date'] == target_date)]
    if area:
        target_data = target_data[target_data['Area'] == area]
    
    if target_data.empty:
        return {
            "error": f"No data found for {city} on {target_date}",
            "city": city,
            "area": area,
            "target_date": target_date
        }
    
    target_total = float(target_data['Total'].sum())
    target_pattern = target_data.groupby('Hour')['Total'].sum().to_dict()
    
    query = f"consumption pattern {city} total {target_total:.0f} MW"
    if area:
        query += f" {area}"
    
    similar_results = rag_pipeline.query_similar_data(query, "energy", k=200)
    
    similar_dates = set()
    for metadata in similar_results['metadatas'][0]:
        date = metadata.get('date')
        if date and date != target_date:
            similar_dates.add(date)
    
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

if __name__ == "__main__":
    print("🚀 Grid Operations MCP Server with LangChain RAG Pipeline")
    print("📊 FAISS vectorstore initialized for embeddings")
    print("🔍 Energy data ingested and ready for queries")
    app.run(transport="stdio")