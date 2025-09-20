from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uuid
import logging
import os
import json
from filelock import FileLock
from google.generativeai import ChatSession, GenerativeModel, configure

# from main2copycopy import graph


logger = logging.getLogger("chat_server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import os
from dotenv import load_dotenv
load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")

if not GENAI_API_KEY:
    raise RuntimeError("Set GENAI_APIKEY environment variable")
configure(api_key=GENAI_API_KEY)


SESSION_DIR = os.getenv("SESSIONSDIR", ".sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


SYSTEM_PROMPT = """
You are an intelligent travel assistant specialized in planning customized travel itineraries. 
Your goal is to gather complete details about the user's trip, understand their preferences, constraints, and generate an optimized day-wise itinerary with suggestions for accommodation, routes, and attractions.

Instructions:
- Ask one question at a time to collect specific trip details.
- Clarify all ambiguous information politely before making suggestions.
- Confirm important inputs such as origin, destination, dates, budget, preferences, and special requirements.
- Use the data you have to generate concise plans, and only answer once you have all necessary details.
- When ready, provide the day-wise itinerary with recommendations, possible routes, travel times, and essential tips.
- If the user wants to change any input, accommodate that in the plan update.

Available Data:
- User's trip origin and destination
- Travel dates and duration
- Budget level (economy, mid-range, luxury)
- Travel companions (solo, couple, family, friends)
- Interests (culture, history, nature, shopping, relaxation, adventure)
- Preferred transport modes (car, train, bus, flight)
- Accommodation preferences and amenities

Remember, interact step-by-step, confirming user's inputs, and produce plans only when information is sufficient.
"""


def _session_file(session_id: str) -> str:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
    return os.path.join(SESSION_DIR, f"{safe}.json")


def _load_session(session_id: str) -> list:
    sf = _session_file(session_id)
    if os.path.exists(sf):
        with open(sf, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_session(session_id: str, conversation: list) -> None:
    sf = _session_file(session_id)
    with open(sf, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    feedback: str | None = None


@app.get("/session/new")
async def new_session():
    sid = uuid.uuid4().hex
    logger.info(f"Generated new session_id={sid}")
    return {"session_id": sid}


@app.post("/chat", response_class=PlainTextResponse)
async def chat(req: ChatRequest, request: Request):
    incoming_sid = (req.session_id or "").strip()
    if not incoming_sid or incoming_sid == "default":
        session_id_to_use = uuid.uuid4().hex
        created_new_sid = True
    else:
        session_id_to_use = incoming_sid
        created_new_sid = False

    lock_file = _session_file(session_id_to_use) + ".lock"
    try:
        with FileLock(lock_file):
            conversation = _load_session(session_id_to_use)

            # Add system prompt at start of new session
            if not conversation:
                conversation.append({"role": "system", "content": SYSTEM_PROMPT})

            # Invoke the graph with latest message
            cfg = {"configurable": {"thread_id": session_id_to_use}}
            delta = {"user_input": req.message}
            if req.feedback:
                delta["user_feedback"] = req.feedback

            # try:
            #     graph_output = graph.invoke(delta, config=cfg)
            #     graph_state = graph.get_state(cfg).values
            #     graph_msgs = graph_state.get("messages") or []
            #     plan_text = graph_state.get("plan")
            # except Exception as e:
            #     logger.warning(f"Graph invocation error for session {session_id_to_use}: {e}")
            #     graph_msgs = []
            #     plan_text = None

            # for gm in graph_msgs:
            #     try:
            #         content = getattr(gm, "content", None) or str(gm)
            #     except Exception:
            #         content = str(gm)
            #     if content:
            #         conversation.append({"role": "assistant", "content": content})

            # if plan_text:
            #     conversation.append({"role": "system", "content": f"Plan: {plan_text}"})

            conversation.append({"role": "user", "content": req.message})

            chat_model = GenerativeModel()
            chat_session = ChatSession(chat_model)

            def conversation_to_prompt(convo):
                parts = []
                for m in convo:
                    r = m.get("role", "user")
                    c = m.get("content", "")
                    parts.append(f"{r}: {c}")
                return "\n".join(parts)

            prompt_text = conversation_to_prompt(conversation)

            response = chat_session.send_message(prompt_text)
            assistant_content = ""
            try:
                assistant_content = response.result.candidates[0].content.parts[0].text
            except Exception:
                assistant_content = str(response)

            assistant_msg = {"role": "assistant", "content": assistant_content}
            conversation.append(assistant_msg)
            _save_session(session_id_to_use, conversation)

    except Exception as e:
        logger.exception(f"Error in /chat for session {session_id_to_use}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


    bot_text = ""
    try:
        bot_text = assistant_msg.get("content", "")
    except Exception as e:
        logger.error(f"Error extracting assistant message content: {e}")

    headers = {"x-session-id": session_id_to_use}

    if created_new_sid:
        logger.info(f"Created new session_id={session_id_to_use} for request from {request.client.host if request.client else 'unknown'}")

    logger.debug(f"Returning response length={len(bot_text)} for session {session_id_to_use}")
    return PlainTextResponse(content=bot_text or "", headers=headers)






