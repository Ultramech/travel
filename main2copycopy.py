from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import TextSplitter
from langchain_community.llms import Ollama
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
import logging
import os,time
import json
from typing import Optional, Dict, Any,List,Tuple

from datetime import datetime, date, time, timedelta
import pytz
import re
import logging
from dateutil import parser as dateparser

# Robust date/time libs (optional)
try:
    import dateparser as _dateparser_pkg
except Exception:
    _dateparser_pkg = None
try:
    import parsedatetime as _parsedatetime_pkg
    _pdt_calendar = _parsedatetime_pkg.Calendar()
except Exception:
    _pdt_calendar = None

# ---------------- Clarifying question templates ----------------
CLARIFY_TEMPLATES = {
    "origin": "Please provide the Origin city or town (e.g., 'Delhi'). Reply with only the city name.",
    "destination": "Please provide the Destination city, town, or region (e.g., 'Ranchi' or 'Jharkhand'). Reply with only the place name.",
    "start_date": "Please provide the start date of travel in YYYY-MM-DD format (example: 2025-12-01). Reply with the date only.",
    "end_date": "Please provide the end date of travel in YYYY-MM-DD format (example: 2025-12-05). Reply with the date only. (If you know start_date and duration, you can give either.)",
    "duration": "Please provide the trip duration in whole days as an integer (example: 3). Reply with a number only (e.g., '4').",
    "preferences": "Please provide travel preferences as comma-separated keywords from (food, nature, adventure, shopping, sightseeing, culture, history, relaxation). Example: 'nature, culture'.",
    "Persons": "Please provide the number of persons travelling as an integer (example: 2). Reply with a number only."
}

def generate_clarifying_question(missing_key: str, parsed_state: dict | None = None) -> str:
    """
    Return a short, exact clarifying prompt for a single field.
    """
    base = CLARIFY_TEMPLATES.get(missing_key)
    if not base:
        return f"Please provide {missing_key} (short answer)."
    if missing_key == "destination" and parsed_state:
        origin = parsed_state.get("origin")
        if origin:
            return base + f" (Trip origin is {origin}.)"
    return base

# ---------------- Logging ----------------
logger = logging.getLogger("trip")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

# ---------------- Time parsing helpers ----------------
USER_TZ = pytz.timezone("Asia/Kolkata")

_time_keywords = {
    "morning": time(9, 0),
    "afternoon": time(15, 0),
    "evening": time(18, 0),
    "night": time(20, 0),
    "tonight": time(20, 0),
    "noon": time(12, 0),
    "midnight": time(0, 0),
    "early morning": time(6, 0),
}

def _extract_time_by_regex(s: str) -> time | None:
    if not s: return None
    s = s.lower()
    for k, t in _time_keywords.items():
        if k in s:
            return t
    m = re.search(r'(\b\d{1,2})(?::|\.)(\d{2})\s*(am|pm)?\b', s)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ampm = m.group(3)
        if ampm:
            if ampm == "pm" and hh < 12: hh += 12
            if ampm == "am" and hh == 12: hh = 0
        try:
            return time(hh % 24, mm)
        except:
            return None
    m2 = re.search(r'\b(\d{1,2})\s*(am|pm)\b', s)
    if m2:
        hh = int(m2.group(1)); ampm = m2.group(2)
        if ampm == "pm" and hh < 12: hh += 12
        if ampm == "am" and hh == 12: hh = 0
        return time(hh % 24, 0)
    return None

def _parse_with_dateparser(text: str, ref_dt: datetime | None = None) -> (date | None, time | None):
    if _dateparser_pkg is None:
        return None, None
    settings = {
        "TIMEZONE": str(USER_TZ),
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": (ref_dt or datetime.now(USER_TZ))
    }
    try:
        dt = _dateparser_pkg.parse(text, settings=settings)
        if dt is None:
            return None, None
        parsed_time = dt.time() if dt.time() != datetime.min.time() else None
        return dt.date(), parsed_time
    except Exception as e:
        logger.debug("dateparser failed: %s", e)
        return None, None

def _parse_with_parsedatetime(text: str, ref_dt: datetime | None = None) -> (date | None, time | None):
    if _pdt_calendar is None:
        return None, None
    try:
        ref = (ref_dt or datetime.now(USER_TZ))
        struct, flags = _pdt_calendar.parseDT(text, sourceTime=ref)
        if not struct:
            return None, None
        parsed_date = struct.date()
        parsed_time = struct.time() if struct.time() != datetime.min.time() else None
        return parsed_date, parsed_time
    except Exception as e:
        logger.debug("parsedatetime failed: %s", e)
        return None, None

def canonicalize_with_llm(text: str) -> (str | None, str | None):
    """
    Fallback: ask the LLM to canonicalize (only used if the main parsers fail).
    Returns (start_date_iso or None, arrival_time 'HH:MM' or None)
    """
    try:
        if 'llm' not in globals():
            return None, None
        prompt = f"""
You are a utility that extracts travel date/time from a short user phrase. Return ONLY JSON with keys:
start_date, end_date, arrival_time.
Rules:
- Emit ISO date YYYY-MM-DD for start_date and end_date when you can.
- Emit arrival_time as HH:MM (24-hour) if a time is present.
- If you cannot determine a field, return empty string "" for it.
User text: \"{text}\"
"""
        resp = llm.invoke([{"role": "user", "content": prompt}])
        content = getattr(resp, "content", None) or (resp.get("content") if isinstance(resp, dict) else None) or str(resp)
        m = re.search(r'(\{.*\})', content, flags=re.S)
        json_text = m.group(1) if m else content
        import json
        data = json.loads(json_text)
        sd = data.get("start_date") or None
        at = data.get("arrival_time") or None
        return sd, at
    except Exception as e:
        logger.debug("LLM canonicalize failed: %s", e)
        return None, None

def parse_date_time(text: str, ref_dt: datetime | None = None) -> (str | None, str | None):
    """
    Robust pipeline to extract date/time from free text.
    Returns (date_iso or None, time_str 'HH:MM' or None)
    """
    if not text:
        return None, None
    ref_dt = ref_dt or datetime.now(USER_TZ)
    text = str(text).strip()

    # 1) explicit iso date + time
    m = re.search(r'(\d{4}-\d{2}-\d{2})[T\s]+(\d{1,2}:\d{2})', text)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            t = datetime.strptime(m.group(2), "%H:%M").time()
            return d.isoformat(), t.strftime("%H:%M")
        except:
            pass

    # 2) dateparser
    d, t = _parse_with_dateparser(text, ref_dt)
    if d or t:
        date_iso = d.isoformat() if d else None
        time_str = t.strftime("%H:%M") if t else None
        return date_iso, time_str

    # 3) parsedatetime
    d, t = _parse_with_parsedatetime(text, ref_dt)
    if d or t:
        return (d.isoformat() if d else None), (t.strftime("%H:%M") if t else None)

    # 4) heuristic regex/time keywords
    t2 = _extract_time_by_regex(text)
    d2 = None
    if re.search(r'\btoday\b', text, flags=re.I):
        d2 = ref_dt.date()
    elif re.search(r'\btomorrow\b', text, flags=re.I):
        d2 = ref_dt.date() + timedelta(days=1)
    elif re.search(r'\bin\s+\d+\s+days?\b', text, flags=re.I):
        m = re.search(r'in\s+(\d+)\s+days?', text, flags=re.I)
        if m:
            d2 = ref_dt.date() + timedelta(days=int(m.group(1)))
    if d2 or t2:
        return (d2.isoformat() if d2 else None), (t2.strftime("%H:%M") if t2 else None)

    # 5) final fallback: LLM canonicalize
    sd, at = canonicalize_with_llm(text)
    return sd, at

# ---------------- normalize_parsed (updated to capture arrival_time) ----------------
def normalize_parsed(parsed: dict) -> dict:
    """Normalize keys & basic types from the LLM parser so merging is stable."""
    if not parsed:
        return {}

    def _clean_str(v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def _to_int(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                return None
        s = str(v)
        m = re.search(r"-?\d+", s)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
        return None

    keys_map = {
        "origin": "origin",
        "destination": "destination",
        "start_date": "start_date",
        "end_date": "end_date",
        "duration": "duration",
        "preferences": "preferences",
        "persons": "Persons",
        "Persons": "Persons",
        "out_of_scope": "out_of_scope",
    }

    out: dict = {}
    for k, v in parsed.items():
        kk = keys_map.get(k, k)

        if kk in {"start_date", "end_date"}:
            s = _clean_str(v)
            if s:
                date_iso, time_str = parse_date_time(s)
                out[kk] = date_iso if date_iso else s
                if kk == "start_date" and time_str:
                    out.setdefault("arrival_time", None)
                    out["arrival_time"] = time_str
            else:
                out[kk] = None

        elif kk == "origin":
            s = _clean_str(v)
            out[kk] = s.title() if s else None

        elif kk == "destination":
            s = _clean_str(v)
            out[kk] = s.title() if s else None

        elif kk == "Persons":
            out[kk] = _to_int(v)

        elif kk == "duration":
            out[kk] = _to_int(v)

        elif kk == "preferences":
            if v is None:
                out[kk] = None
            elif isinstance(v, list):
                cleaned = [str(x).strip() for x in v if str(x).strip()]
                out[kk] = ", ".join(dict.fromkeys(cleaned)).title() if cleaned else None
            else:
                s = str(v).strip()
                if "," in s:
                    items = [i.strip() for i in s.split(",") if i.strip()]
                    out[kk] = ", ".join(dict.fromkeys(items)).title() if items else None
                else:
                    out[kk] = s.title() if s else None

        elif kk == "out_of_scope":
            b = _to_bool(v)
            out[kk] = b if b is not None else False

        else:
            out[kk] = _clean_str(v) if isinstance(v, str) else v

    return out

# ---------------- initial state ----------------
def initial_state():
    return {
        "user_input": None,
        "messages": [],
        "parsed": {},
        "arrival_time": None,
        "transportation_summary": None,
        "weather_data": [],
        "hotels_info": None,
        "user_feedback": None,
        "human_review_result": None,
        "llm_routing_decision": None,
        "plan": None,
    }

# ---------------- Google Gemini client & contents builder ----------------
import google.generativeai as genai
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("Please set GENAI_API_KEY environment variable")
genai_client = genai.configure(GENAI_API_KEY)

# ---------------- convert dict messages to LangChain messages ----------------
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def to_langchain_messages(dict_msgs):
    out = []
    for m in dict_msgs:
        if not isinstance(m, dict):
            # assume it's already a message object
            out.append(m)
            continue
        role = m.get("role")
        text = m.get("text") or m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out


def build_contents_from_state(state: dict, max_turns: int = 40):
    msgs = state.get("messages") or []
    norm = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or "user"
            text = m.get("text") or m.get("content") or ""
        else:
            role = getattr(m, "type", "user")
            text = getattr(m, "content", str(m))
        if role == "human":
            role = "user"
        norm.append({"role": role, "text": text})
    trimmed = norm[-max_turns:]
    if not trimmed or trimmed[0].get("role") != "system":
        trimmed.insert(0, {"role": "system", "text": "You are a helpful travel assistant."})
    return trimmed

# ---------------- generate_daywise_plan (strong prompt to avoid origin/destination confusion) ----------------
def generate_daywise_plan(query: str, origin: str, destination: str, interest: str,
                          weather_data: list, duration: str, arrival_time: str,
                          arrival_date: str, hotel_info: dict, persons: int,
                          feedback: str="") -> str:

    from langchain_community.document_loaders import WebBaseLoader
    # local import guard - optional
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import Chroma
    except Exception:
        HuggingFaceEmbeddings = None
        RetrievalQA = None
        Chroma = None

    country = "India"

    weather_context = []
    for day in weather_data:
        weather_context.append(
            f"{day.get('day_name','')}: {day.get('condition','')} ({day.get('min_temp','?')}°C to {day.get('max_temp','?')}°C), "
            f"Rain: {day.get('rain_chance',0)}%, Humidity: {day.get('humidity',0)}%"
        )
    weather_summary = "\n".join(weather_context)
    import json
    from langchain.docstore.document import Document

    try:
        with open("festival_details.json", "r", encoding="utf-8") as f:
            jharkhand_data = json.load(f)
    except Exception:
        jharkhand_data = {"festivals": {"tribal_regional": [], "pilgrimage_religious": []}, "distances": {}, "offbeat_locations": []}

    json_docs = []

    for fest in jharkhand_data.get("festivals", {}).get("tribal_regional", []):
        text = f"""
        Festival: {fest.get('name','')}
        Time: {fest.get('time', '')}
        Duration: {fest.get('duration', '')}
        Rituals: {', '.join(fest.get('rituals', []))}
        Significance: {fest.get('significance', '')}
        Celebrated by: {', '.join(fest.get('celebrated_by', []))}
        """
        json_docs.append(Document(page_content=text, metadata={"type": "festival", "category": "tribal"}))

    for fest in jharkhand_data.get("festivals", {}).get("pilgrimage_religious", []):
        text = f"""
        Festival: {fest.get('name','')}
        Location: {fest.get('location', '')}
        Time: {fest.get('time', '')}
        Rituals: {', '.join(fest.get('rituals', []))}
        Significance: {fest.get('significance', '')}
        """
        json_docs.append(Document(page_content=text, metadata={"type": "festival", "category": "religious"}))

    for city, connections in jharkhand_data.get("distances", {}).items():
        for dest, d in connections.items():
            text = f"Distance from {city} to {dest}: Road {d.get('road_km','?')} km, Aerial {d.get('aerial_km','?')} km"
            json_docs.append(Document(page_content=text, metadata={"type": "distance"}))

    for place in jharkhand_data.get("offbeat_locations", []):
        text = f"""
        Offbeat Location: {place.get('name','')} at {place.get('location','')}
        Features: {', '.join(place.get('features', []))}
        Significance: {place.get('significance','')}
        """
        json_docs.append(Document(page_content=text, metadata={"type": "offbeat"}))

    urls = [
        f"https://en.wikivoyage.org/wiki/{destination}_({country})",
        f"https://en.wikivoyage.org/wiki/{destination}",
        "https://en.wikivoyage.org/wiki/Ranchi",
        "https://en.wikivoyage.org/wiki/Bokaro_Steel_City",
        "https://en.wikivoyage.org/wiki/Chaibasa",
        "https://en.wikivoyage.org/wiki/Daltonganj",
        "https://en.wikivoyage.org/wiki/Deoghar",
        "https://en.wikivoyage.org/wiki/Dhanbad",
        "https://en.wikivoyage.org/wiki/Ghatshila",
        "https://en.wikivoyage.org/wiki/Giridih",
        "https://en.wikivoyage.org/wiki/Hazaribag",
        "https://en.wikivoyage.org/wiki/Jamshedpur",
        "https://en.wikivoyage.org/wiki/Madhupur",
        "https://en.wikivoyage.org/wiki/Maithon",
        "https://en.wikivoyage.org/wiki/Massanjore",
        "https://en.wikivoyage.org/wiki/Betla_National_Park",
        "https://en.wikivoyage.org/wiki/Hazaribagh_National_Park",
        "https://en.wikipedia.org/wiki/Panchet_Dam",
        "https://en.wikivoyage.org/wiki/Parasnath_Hills",
    ]

    docs = []
    for url in urls:
        try:
            loaded = WebBaseLoader(url).load()
            docs.extend(loaded)
            print(f"✅ Loaded {url}")
        except Exception as e:
            print(f"⚠ Could not load {url}: {e}")
    all_docs = docs + json_docs

    # embeddings + qa chain only if libs available
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        documents = text_splitter.split_documents(all_docs)
        # lazy import / guard
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        from langchain_community.vectorstores import Chroma
        db = Chroma.from_documents(documents, embeddings)

        retriever = db.as_retriever(search_kwargs={"k": 5})
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        GROQ_API_KEY=os.getenv('GROQ_API_KEY')

        llm_inner = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,  # <<-- replace with valid key
            base_url="https://api.groq.com/openai/v1"
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm_inner, retriever=retriever)

        # ------------------ Strong Prompt: facts + critical constraints ------------------
        full_query = f"""
You are a professional travel planner. Create a detailed {duration}-day itinerary.

FACTS:
- ORIGIN (departure city, starting point only): {origin}
- DESTINATION (trip focus): {destination}
- Arrival in {destination} on {arrival_date} at {arrival_time}
- User interests: {interest}
- Number of persons: {persons}
- Weather forecast (per day): 
{weather_summary}
- Hotel Info: {hotel_info}
- Feedback: {feedback}

CRITICAL INSTRUCTIONS (MUST FOLLOW EXACTLY):
1) The ITINERARY MUST BE CENTERED ON THE DESTINATION: {destination}.
   - You MUST treat {origin} ({origin}) strictly as the departure/starting point only.
   - Do NOT include sightseeing, day plans, or overnight stays in {origin} unless the user explicitly requests activities in the origin city.
2) If {destination} is a STATE/REGION, include multiple cities within that region and make a progressive route (no repeated hub unless user asked).
3) Provide Day-by-day structure: Day 1, Day 2, ... For each day include Morning / Afternoon / Evening / Night.
4) Clearly mark intercity travel legs (mode, approximate departure/arrival times, and travel time) and overnight stay city/hotel for each night.
5) Use weather info to recommend weather-appropriate activities.
6) Provide an explicit budget breakdown at the end (hotels, intercity travel, local transport, food, attractions, misc) and a single line:
   Estimated_budget: <amount in INR>
7) If any required info is missing (dates, persons), mention it as a TODO at the top and make reasonable assumptions, and explicitly state assumptions made.
8) NEVER swap origin and destination in any text, headings, or schedules.

OUTPUT FORMAT:
- Start with a one-line summary: "Trip: {origin} → {destination} | {duration} days | {persons} persons"
- Then the day-by-day itinerary per the rules above.
- End with "Estimated_budget: <amount>"

Use wiki/retrieved docs to enrich suggestions and local insights. Keep the output concise but complete.
"""

        # perform retrieval-augmented planning call
        response = qa_chain.invoke(full_query)
        # extract text result (support multiple return shapes)
        result_text = response.get("result") if isinstance(response, dict) else getattr(response, "content", str(response))
    except Exception as e:
        logger.debug("QA chain failed: %s", e)
        result_text = f"(Could not generate full WikiVoyage plan programmatically: {e})\nPlease create a plan from {origin} to {destination}."

    itinerary = f"🧭 {duration}-Day Itinerary for {origin} → {destination}\n\n" + result_text
    return itinerary

# ---------------- Hotel SERP API ----------------
def search_hotels_serpapi2(city: str, checkin: str, checkout: str, duration: int, feedback: str, persons: int = 1) -> str:
    import requests
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    cleaned_feedback = feedback.lower().replace("feedback", "").strip()

    params = {
        "engine": "google_hotels",
        "q": f"hotels in {city}",
        "check_in_date": checkin,
        "check_out_date": checkout,
        "adults": persons,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    if cleaned_feedback:
        params["q"] += f" {cleaned_feedback}"

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=8)
    except Exception as e:
        return f"❌ SerpAPI request failed: {e}"

    print("🔍 SerpAPI Request URL:", resp.url)

    if resp.status_code != 200:
        return f"❌ Error: HTTP {resp.status_code} - {resp.text}"

    data = resp.json()
    hotels = data.get("properties") or data.get("hotels")
    if not hotels:
        return "🚫 No hotels found—Google may not cover this area."

    top = hotels[:7]
    out = f"🏨 Top {len(top)} Hotels in {city} from {checkin} to {checkout}:\n\n"
    for h in top:
        name = h.get("name", "Unknown")
        price = h.get("rate_per_night", {}).get("lowest", "N/A")
        rating = h.get("overall_rating", h.get("rating", "N/A"))
        link = h.get("link", "")

        try:
            total_price = float(price) * duration if price != "N/A" else "N/A"
        except:
            total_price = "N/A"

        out += f"🔹 {name}\n💰 {total_price} for {persons} persons | ⭐ {rating}\n🔗 {link}\n\n"

    return out

# ---------------- extract_trip_fields (updated prompt and LLM usage) ----------------
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
import os

import pandas as pd
import json

def extract_trip_fields(user_input: str, known_state: dict | None = None):
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from langchain.schema import SystemMessage, HumanMessage

    response_schemas = [
        ResponseSchema(name="origin", description="Starting location (e.g., Delhi)"),
        ResponseSchema(name="destination", description="Trip destination (e.g., Paris or Jharkhand)"),
        ResponseSchema(name="start_date", description="Start date of trip (YYYY-MM-DD)"),
        ResponseSchema(name="end_date", description="End date of trip (YYYY-MM-DD)"),
        ResponseSchema(name="duration", description="Duration of trip (in days)"),
        ResponseSchema(name="preferences", description="List of interests like food, nature, adventure, shopping, sightseeing, culture and history"),
        ResponseSchema(name="Persons", description="Number of persons going on the trip"),
        ResponseSchema(name="out_of_scope", description="Boolean true/false. True if the text is not related to travel at all.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    known_state_text = json.dumps(known_state, indent=2) if known_state else "None yet"

    system_instructions = """
    You are a travel planner assistant whose job is to EXTRACT STRUCTURED TRIP FIELDS and ALWAYS RETURN VALID JSON.
    Return exactly the fields: origin, destination, start_date, end_date, duration, preferences, Persons, out_of_scope.
    Rules (strict):
    - Dates must be in ISO format YYYY-MM-DD (example: 2025-12-01). If you have a start_date and duration, compute end_date = start_date + duration days and fill it.
    - duration and Persons must be integers (e.g., 3). If user says "3 days" or "for 2 people", extract the integer.
    - preferences should be a comma-separated lowercase list drawn from: food, nature, adventure, shopping, sightseeing, culture, history, relaxation (if other, keep as free text but comma-separated).
    - origin and destination should be short place names or city names (e.g., "Delhi", "Paris", "Jharkhand"). If a state/region is given, that's OK.
    - out_of_scope must be true ONLY if the text is clearly unrelated to travel.
    - If uncertain, set out_of_scope: false and try to extract any partial fields.
    - If the user mentions relative phrases like "today", "tomorrow", "tonight at 6pm", attempt to canonicalize to YYYY-MM-DD and HH:MM for start_date and arrival_time respectively; if unsure, set fields to null.
    - ALWAYS include all keys. Use null for unknown values.
    Return only JSON matching those keys.
    """

    prompt = PromptTemplate(
        template="""
Extract structured trip information from the text below and preserve any previously-known fields where applicable:
Known state (previous fields): {known_state}

{format_instructions}

User text:
{input_text}

IMPORTANT: If you extract a date, format MUST be YYYY-MM-DD.
If computing end_date from start_date + duration, supply end_date in YYYY-MM-DD.
If you cannot parse a field, set it to null, but never omit keys.
""",
        input_variables=["input_text", "known_state"],
        partial_variables={"format_instructions": format_instructions},
    )

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=prompt.format(input_text=user_input, known_state=known_state_text)),
    ]

    from langchain_openai import ChatOpenAI
    GROQ_API_KEY=os.getnev("GROQ_API_KEY")
    # instantiate llm locally so canonicalize_with_llm can also use it via global llm
    global llm
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,  # <<-- replace with valid key
        base_url="https://api.groq.com/openai/v1"
    )

    response = llm.invoke(messages)
    raw_content = getattr(response, "content", None) or str(response)
    print("Raw LLM Output:\n", raw_content)

    parsed = output_parser.parse(raw_content)

    if parsed.get("out_of_scope") is True:
        return {
            "messages": [
                "❌ This query looks unrelated to travel. "
                "I am a travel planner and can only help with trip planning. "
                "👉 Please tell me details like your origin, destination, travel dates, interests, and number of persons."
            ],
            "llm_routing_decision": "END"
        }

    return {
        "parsed": parsed,
        "messages": [f"✅ Extracted trip details: {parsed}"],
        "llm_routing_decision": "Fill_Missing_Info"
    }

# ---------------- State type and graph basics ----------------
from typing import List, Dict, Any
from langgraph.graph.message import add_messages
from langchain.schema import BaseMessage

class State(TypedDict):
    user_input: Annotated[Optional[str], lambda old, new: new or old]
    messages: Annotated[List[BaseMessage], add_messages]
    parsed: Annotated[Dict[str, Any], lambda old, new: {**(old or {}), **(new or {})}]
    arrival_time: Annotated[Optional[str], lambda old, new: new or old]
    transportation_summary: Annotated[Optional[str], lambda old, new: new or old]
    weather_data: Annotated[list, lambda old, new: (old or []) + (new or [])]
    hotels_info: Annotated[Optional[str], lambda old, new: new or old]
    user_feedback: Annotated[Optional[str], lambda old, new: new or old]
    human_review_result: Annotated[Optional[str], lambda old, new: new or old]
    llm_routing_decision: Annotated[Optional[str], lambda old, new: new or old]
    plan: Annotated[Optional[str], lambda old, new: new or old]

graph_builder = StateGraph(State)

# ---------------- Feedback router (unchanged) ----------------
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

def llm_feedback_router(state: State) -> State:
    """Handle user feedback and route to appropriate node"""
    plan = state.get("plan", "⚠ No plan available.")
    feedback = state.get("user_feedback", "")

    if not feedback:
        return {
            **state,
            "messages": state.get("messages", []) + [
                AIMessage(content="No feedback received. Keeping the current plan.")
            ],
            "llm_routing_decision": "END"
        }

    prompt = f"""
You are a travel assistant. A user has given the following feedback on their travel plan:

"{feedback}"

Decide which part needs to be updated:
- "Hotels" if they want to change hotels or accommodation
- "DayWise_Plan" if they want to change the itinerary, activities, or schedule
- "END" if no change is required or feedback is unclear

Just respond with the name of the component. Don't include anything else.
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    decision = result.content.strip()
    valid_options = {"Hotels", "DayWise_Plan", "END"}
    if decision not in valid_options:
        decision = "END"

    return {
        **state,
        "llm_routing_decision": decision,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"📝 Feedback processed. Routing to: {decision}")
        ]
    }

# %% Weather node
import requests
from langchain.tools import Tool

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather_forecast_node(state: State) -> State:
    parsed = state.get("parsed", {})
    messages = state.get("messages", []) or []

    try:
        city = parsed.get("destination")
        start_date_str = parsed.get("start_date")
        num_days = int(parsed.get("duration") or 3)

        if not city or not start_date_str:
            raise ValueError("City or start date missing in parsed state")

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        today = datetime.today().date()

        results = []
        weather_data = []

        for i in range(min(num_days, 14)):
            target_date = start_date + timedelta(days=i)
            endpoint = "forecast.json" if target_date >= today else "history.json"
            url = f"http://api.weatherapi.com/v1/{endpoint}"
            params = {
                "key": WEATHER_API_KEY,
                "q": city,
                "dt": target_date.strftime("%Y-%m-%d")
            }

            try:
                res = requests.get(url, params=params, timeout=5)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RequestException as re:
                return {
                    "messages": messages + [AIMessage(content=f"❌ Weather API error: {str(re)}")],
                    "parsed": parsed
                }

            if "error" in data:
                return {
                    "messages": messages + [AIMessage(content=f"❌ API Error on {target_date}: {data['error']['message']}")],
                    "parsed": parsed
                }

            forecast_days = data.get("forecast", {}).get("forecastday", [])
            if not forecast_days:
                return {
                    "messages": messages + [AIMessage(content=f"❌ No forecast data for {target_date}")],
                    "parsed": parsed
                }

            day_info = forecast_days[0]['day']
            condition = day_info['condition']['text']
            max_temp = day_info['maxtemp_c']
            min_temp = day_info['mintemp_c']

            results.append(f"📅 {target_date.strftime('%b %d, %Y')}: {condition}, 🌡 {min_temp}°C to {max_temp}°C")

            weather_data.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "day_name": target_date.strftime("%A"),
                "condition": condition,
                "max_temp": max_temp,
                "min_temp": min_temp,
                "avg_temp": day_info.get('avgtemp_c', (max_temp + min_temp) / 2),
                "rain_chance": day_info.get('daily_chance_of_rain', 0),
                "snow_chance": day_info.get('daily_chance_of_snow', 0),
                "humidity": day_info.get('avghumidity', 0)
            })

        forecast_message = f"📍 Weather Forecast for {city}\n\n" + "\n".join(results)
        return {
            "messages": messages + [AIMessage(content=forecast_message)],
            "weather_data": weather_data
        }

    except Exception as e:
        return {
            "messages": messages + [AIMessage(content=f"❌ Weather Forecast Node Error: {str(e)}")],
            "weather_data": []
        }

# %% Daywise plan node
def wikivoyage_daywise_plan(state: State) -> State:
    user_input = state.get("messages", [])[-1].content if state.get("messages") else ""
    tour_info = []

    try:
        parsed = state['parsed']
        destination = parsed["destination"]
        origin = parsed.get("origin", "")
        country = "India"
        interest = parsed.get("preferences", "sightseeing")
        weather_data = state.get("weather_data", [])
        duration = parsed.get("duration", "3")
        arrival_time = parsed.get("arrival_time") or state.get("arrival_time") or "14:00"
        arrival_date = parsed.get("start_date")
        hotel_info = state.get("hotels_info")
        persons = int(parsed.get("Persons") or parsed.get("persons") or 1)
        feedback = state.get("human_feedback", "")
    except Exception as e:
        return {
            "messages": [*state.get("messages", []), AIMessage(content=f"❌ Failed to extract trip fields: {e}")]
        }

    try:
        plan = generate_daywise_plan(user_input, origin, destination, interest, weather_data, duration, arrival_time, arrival_date, hotel_info, persons, feedback)
        tour_info.append(f"🧭 Itinerary:\n{plan}")
    except Exception as e:
        tour_info.append(f"❌ WikiVoyage Tool failed: {e}")

    combined_output = "\n\n".join(tour_info)
    return {
        "messages": [*state.get("messages", []), AIMessage(content=combined_output)],
        "plan": combined_output
    }

# %% Hotels node
def search_hotels_serpapi(state: State) -> State:
    try:
        parsed = state['parsed']
        feedback = state.get("user_feedback", "")
        hotels = search_hotels_serpapi2(parsed["destination"], parsed["start_date"], parsed["end_date"], parsed["duration"], feedback, parsed["Persons"])
        hotel_info = f"🏨 Hotels in {parsed['destination']}:\n{hotels}"
        return {
            "messages": [*state.get("messages", []), AIMessage(content=hotel_info)],
            "parsed": parsed,
            "decision": state.get("decision"),
            "hotels_info": hotel_info
        }
    except Exception as e:
        error_msg = f"❌ Hotels Tool failed: {e}"
        return {
            "messages": [*state.get("messages", []), AIMessage(content=error_msg)],
            "decision": state.get("decision")
        }

# %% detail_extraction_node (conservative merge, uses normalize_parsed)


def detail_extraction_node(state: State) -> State:
    """
    Node to extract structured trip fields from user input.
    - Uses extract_trip_fields with known_state for context.
    - Merges non-None fields into existing parsed state (prevents overwriting with nulls).
    - Preserves 'out_of_scope' if set True.
    - Adds any messages returned by parser.
    """

    user_input = state.get("user_input", "")
    parser_result = extract_trip_fields(user_input, known_state=state.get("parsed", {}))

    # Handle parser output safely
    parsed_from_llm = {}
    if isinstance(parser_result, dict):
        parsed_from_llm = parser_result.get("parsed", {}) or {}

    # Normalize parsed (define normalize_parsed to clean types: e.g., string→int)
    parsed_from_llm = normalize_parsed(parsed_from_llm)

    # Merge into existing parsed state
    existing = state.get("parsed", {}) or {}
    merged = dict(existing)  # copy current state

    for k, v in parsed_from_llm.items():
        if v is not None:  # only update if not None
            merged[k] = v

    # Explicitly preserve out_of_scope=True
    if parsed_from_llm.get("out_of_scope") is True:
        merged["out_of_scope"] = True

    # Messages
    messages_in = state.get("messages", []) or []
    new_msgs = []
    if isinstance(parser_result, dict):
        new_msgs = [AIMessage(content=msg) for msg in parser_result.get("messages", []) or []]

    messages = messages_in + new_msgs

    # Routing decision
    routing = "Fill_Missing_Info"
    if isinstance(parser_result, dict):
        routing = parser_result.get("llm_routing_decision", "Fill_Missing_Info")

    if merged.get("out_of_scope") is True:
        routing = "END"

    # Debug logging
    logger.debug("detail_extraction_node - parsed_from_llm: %s", parsed_from_llm)
    logger.debug("detail_extraction_node - merged_parsed: %s", merged)

    return {
        "user_input": user_input,
        "parsed": merged,
        "messages": messages,
        "llm_routing_decision": routing,
    }


# %% fill_missing_info_node (improved)
def is_valid_date(s: str) -> bool:
    try:
        d = dateparser.parse(s, fuzzy=False)
        return True
    except Exception:
        return False

def try_parse_date_from_text(s: str):
    try:
        d = dateparser.parse(s, fuzzy=True)
        return d.date().isoformat()
    except Exception:
        return None

def try_parse_int_from_text(s: str):
    m = re.search(r"-?\d+", str(s))
    return int(m.group(0)) if m else None

def fill_missing_info_node(state: State) -> State:
    """
    Deterministic missing-info node:
    - attempt to parse missing fields from user_input locally
    - if still missing, ask a single precise question and set routing to END so frontend collects it
    - if everything filled, route forward to Weather_Forecast
    """
    current_parsed = state.get("parsed", {}) or {}
    user_input = (state.get("user_input") or "").strip()

    newly = {}
    if user_input:
        sd_iso, time_str = parse_date_time(user_input)
        if sd_iso and not current_parsed.get("start_date"):
            newly["start_date"] = sd_iso
            if time_str:
                newly["arrival_time"] = time_str

        ed_iso, _ = parse_date_time(user_input)
        if ed_iso and not current_parsed.get("end_date") and ed_iso != newly.get("start_date"):
            newly["end_date"] = ed_iso

        if ("duration" not in current_parsed or not current_parsed.get("duration")) and any(w in user_input.lower() for w in ["day", "days"]):
            i = try_parse_int_from_text(user_input)
            if i is not None:
                newly["duration"] = int(i)

        if ("Persons" not in current_parsed or not current_parsed.get("Persons")) and any(w in user_input.lower() for w in ["person", "people", "persons", "adults"]):
            p = try_parse_int_from_text(user_input)
            if p is not None:
                newly["Persons"] = int(p)

        # Short non-date replies: try to infer origin/destination but conservatively
        if ("origin" not in current_parsed or not current_parsed.get("origin")) and len(user_input.split()) <= 3 and ',' not in user_input and not is_valid_date(user_input):
            if not newly.get("start_date") and not newly.get("duration"):
                newly["origin"] = user_input.title()

        if ("destination" not in current_parsed or not current_parsed.get("destination")) and len(user_input.split()) <= 3 and ',' not in user_input and not is_valid_date(user_input):
            if not newly.get("start_date") and not newly.get("duration"):
                newly["destination"] = user_input.title()

    merged = {**current_parsed}
    for k, v in newly.items():
        if v is not None:
            merged[k] = v

    # Compute end_date if start_date + duration available and end_date missing
    try:
        if merged.get("start_date") and merged.get("duration") and not merged.get("end_date"):
            sd = datetime.strptime(merged["start_date"], "%Y-%m-%d").date()
            merged["end_date"] = (sd + timedelta(days=int(merged["duration"]) - 1)).isoformat()
    except Exception as e:
        logger.debug("Could not compute end_date: %s", e)

    required = ["origin", "destination", "start_date", "end_date", "duration", "preferences", "Persons"]
    missing = [k for k in required if merged.get(k) in [None, "", []]]

    logger.debug("fill_missing_info_node - newly parsed: %s", newly)
    logger.debug("fill_missing_info_node - merged: %s", merged)
    logger.debug("fill_missing_info_node - missing: %s", missing)

    if not missing:
        return {
            **state,
            "parsed": merged,
            "llm_routing_decision": "Weather_Forecast",
            "messages": state.get("messages", [])
        }

    ask_key = missing[0]
    question = generate_clarifying_question(ask_key, parsed_state=merged)

    return {
        **state,
        "parsed": merged,
        "messages": state.get("messages", []) + [AIMessage(content=f"❓ {question}")],
        "llm_routing_decision": "END"
    }

# %% route_missing_info
def route_missing_info(state: State) -> str:
    parsed = state.get("parsed", {})
    required_keys = ["origin", "destination", "start_date", "end_date", "duration", "preferences", "Persons"]
    missing = [k for k in required_keys if parsed.get(k) in [None, ""]]
    if missing:
        return "END"
    else:
        return "Weather_Forecast"

# %% Graph builder compile & edges
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import HumanMessage

memory = MemorySaver()
graph_builder = StateGraph(State)

graph_builder.add_node("Detail_Extraction", detail_extraction_node)
graph_builder.add_node("Weather_Forecast", get_weather_forecast_node)
graph_builder.add_node("Hotels", search_hotels_serpapi)
graph_builder.add_node("DayWise_Plan", wikivoyage_daywise_plan)
graph_builder.add_node("User_Feedback", llm_feedback_router)
graph_builder.add_node("Fill_Missing_Info", fill_missing_info_node)

graph_builder.add_edge(START, "Detail_Extraction")
graph_builder.add_edge("Weather_Forecast", "Hotels")
graph_builder.add_edge("Hotels", "DayWise_Plan")
graph_builder.add_edge("DayWise_Plan", "User_Feedback")

graph_builder.add_conditional_edges(
    "Detail_Extraction",
    lambda state: state.get("llm_routing_decision", "Fill_Missing_Info"),
    {
        "END": END,
        "Fill_Missing_Info": "Fill_Missing_Info"
    }
)

graph_builder.add_conditional_edges(
    "Fill_Missing_Info",
    lambda state: state.get("llm_routing_decision", "END"),
    {
        "END": END,
        "Weather_Forecast": "Weather_Forecast"
    }
)
# use MemorySaver (in-memory checkpoint) and compile graph
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
checkpointer = memory
checkpointer.default_config = {"configurable": {"thread_id": "default_thread_id"}}
graph = graph_builder.compile(checkpointer=checkpointer)

