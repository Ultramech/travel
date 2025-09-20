# import streamlit as st
# import requests

# # --- Configuration ---
# API_URL = "https://travel-2-5g7m.onrender.com/chat"  # Replace with your FastAPI endpoint

# st.set_page_config(page_title="Chatbot UI", page_icon="🤖", layout="centered")

# st.title("🤖 Chatbot")
# st.markdown("Type your message and get a response from the AI chatbot.")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # --- User Input ---
# user_input = st.text_input("You:", key="input", placeholder="Type a message and press Enter")

# if st.button("Send") or user_input:
#     if user_input.strip() != "":
#         # Show user message
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         try:
#             # Send message to FastAPI
#             response = requests.post(API_URL, json={"message": user_input})
#             data = response.json()
#             bot_message = data.get("response", "Sorry, no response from server.")
#         except Exception as e:
#             bot_message = f"Error: {str(e)}"

#         # Show bot response
#         st.session_state.messages.append({"role": "bot", "content": bot_message})
#         st.session_state.input = ""  # clear input box

# # --- Display Chat History ---
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"**You:** {msg['content']}")
#     else:
#         st.markdown(f"**Bot:** {msg['content']}")
# streamlit_app.py
import streamlit as st
import requests

# API config (update if your endpoint changes)
API_URL = "https://travel-2-5g7m.onrender.com"

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


# --- Session state ---
if "session_id" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/session/new")
        resp.raise_for_status()
        st.session_state.session_id = resp.json().get("session_id", "default")
    except Exception as e:
        st.error(f"Could not create session: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input" not in st.session_state:
    st.session_state.input = ""


# --- CSS Styles ---
st.markdown("""
<style>
body, .main {
    background: #181c23;
}
h1, h2, h3, h4, h5, h6 { color: #fff!important; }
.header-bar {
    color: #fff;
    font-weight: bold;
    font-size: 2.1rem;
    margin-top: 10px;
    margin-bottom: 0;
    letter-spacing: .03em;
}
.planner-desc {
    color: #b2c1d1;
    margin-bottom: 7px;
    font-size: 1.1rem;
}
.info-bar {
    background: #213048;
    color: #b6d7fa;
    border-radius: 7px;
    font-size: .98rem;
    padding: 7px 12px;
    margin-bottom: 20px;
    margin-top: 2px;
}
.section-title {
    font-weight: bold;
    font-size: 1.18rem;
    margin-top: 12px;
    margin-bottom: 6px;
}
.popular-container {
    display:flex;
    gap: 2vw;
    margin-bottom: 17px;
}
.popular-card {
    flex:1;
    background: #242731;
    border-radius: 13px;
    overflow: hidden;
    box-shadow: 0 3px 18px #050d1e30;
    min-width: 240px;
    max-width: 320px;
    transition: transform .16s;
}
.popular-card:hover {
    transform: scale(1.017);
}
.popular-image {
    width: 100%;
    aspect-ratio: 18/10;
    object-fit: cover;
    border-bottom: 2px solid #1b1f27;
}
.popular-caption {
    color: #d5e9fa;
    font-size: .95rem;
    text-align: center;
    padding: 8px 0 10px 0;
}
.testimonial-row {
    display: flex;
    gap: 1vw;
    width: 100%;
}
.testimonial {
    background: #22252c;
    border-radius: 8px;
    padding: 20px 17px 14px 17px;
    color: #e6eaff;
    font-size: 1.08em;
    margin-bottom: 10px;
    box-shadow: 0 1px 9px #07102724;
    width: 100%;
}
.testimonial .author {
    color: #7ccaf7;
    font-size: .96em;
    margin-top: 8px;
    display: block;
}
#chat-container {
    min-height: 180px;
    max-height: 220px;
    height: 22vh;
    overflow-y: auto;
    margin: 0 0 0 0;
    background: transparent;
    border-radius: 10px;
    padding: 10px;
    box-shadow: none;
    border: none;
}
.user-msg, .bot-msg {
    max-width: 72vw;
    font-size: 1.14rem;
    border-radius: 15px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 8px #0002;
}
.user-msg {
    background: #18977e;
    color: #fff;
    padding: 13px 16px;
    margin-left: auto;
    border-bottom-right-radius: 7px;
}
.bot-msg {
    background: #232842;
    color: #aef9e3;
    padding: 13px 16px;
    border-bottom-left-radius: 7px;
}
.icon {
    width: 26px; height: 26px; border-radius: 50%; margin: 0 8px;
    background-size: contain; background-repeat: no-repeat; background-position: center;
}
.bot-icon {
    background-image: url('https://img.icons8.com/ios-filled/50/ffffff/robot-2.png');
}
.user-icon {
    background-image: url('https://img.icons8.com/ios-filled/50/ffffff/user-male-circle.png');
}
.stTextInput > div > div > input {
    font-size: 1.12rem;
    border-radius: 22px !important;
    height: 44px !important;
    padding-left: 22px !important;
    padding-right: 18px !important;
    background-color: #181b1f !important;
    border: 2px solid #23262b !important;
    box-shadow: none !important;
    color: #f7f7fa !important;
    outline: none !important;
    transition: border 0.2s;
}
.stTextInput > div > div > input:focus {
    border: 2px solid #38bfa7 !important;
}
.stTextInput > div > div > input::placeholder {
    color: #888ea6 !important;
    font-style: normal !important;
    opacity: 1 !important;
}
.stTextInput { box-shadow: none!important; background: none!important; }
</style>
""", unsafe_allow_html=True)



# --- HEADER and above fold ---
st.markdown('<div class="header-bar">✈️ AI Travel Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="planner-desc">Plan your perfect trip with our AI-powered travel assistant. Get personalized recommendations for transportation, accommodation, and activities based on your preferences.</div>', unsafe_allow_html=True)

# --- POPULAR DESTINATIONS ---
st.markdown('<div class="section-title">Popular Destinations</div>', unsafe_allow_html=True)
st.markdown("""
<div class="popular-container">
  <div class="popular-card">
    <img src="https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=600&q=80" class="popular-image"/>
    <div class="popular-caption">Goa · Beaches & Nightlife</div>
  </div>
  <div class="popular-card">
    <img src="https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=600&q=80" class="popular-image"/>
    <div class="popular-caption">Kashmir · Beauty</div>
  </div>
  <div class="popular-card">
    <img src="https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=600&q=80" class="popular-image"/>
    <div class="popular-caption">Kerala · Backwaters & Nature</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- TESTIMONIALS ---
# st.markdown('<div class="section-title" style="margin-top:23px;">What Our Users Say</div>', unsafe_allow_html=True)
# st.markdown("""
# <div class="testimonial-row">
#     <div class="testimonial">
#         "The AI Travel Planner saved me hours of research! The itinerary was perfectly tailored to my interests."
#         <span class="author">– Priya S., Bangalore</span>
#     </div>
#     <div class="testimonial">
#         "I never would have found those hidden gems without this tool. The transportation recommendations were spot on!"
#         <span class="author">– Rajiv M., Delhi</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown('<hr style="border-top:1px solid #253241; margin-bottom:15px; margin-top:24px;">', unsafe_allow_html=True)


# --- CHAT UI ---
st.markdown('<div class="section-title" style="margin-bottom:9px;">Chat with AI Travel Assistant</div>', unsafe_allow_html=True)
chat_container = st.container()

def send_message():
    user_message = st.session_state.input.strip()
    if not user_message:
        return
    st.session_state.messages.append({"role": "user", "content": user_message})
    try:
        payload = {"session_id": st.session_state.session_id, "message": user_message}
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Error contacting FastAPI: {e}"
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.session_state.input = ""

st.text_input(
    "Type your message...",
    key="input",
    on_change=send_message,
    placeholder="Enter your message and hit Enter...",
    label_visibility="collapsed",
)

with chat_container:
    st.markdown('<div id="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-msg">
                    <div class="msg-text">{msg['content']}</div>
                    <div class="icon user-icon" title="User"></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="bot-msg">
                    <div class="icon bot-icon" title="Bot"></div>
                    <div class="msg-text">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.components.v1.html(
    """
    <script>
        const chatContainer = window.parent.document.querySelector('#chat-container');
        if(chatContainer){
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
    """, height=0, width=0,
)
st.markdown('<div class="section-title" style="margin-top:23px;">What Our Users Say</div>', unsafe_allow_html=True)
st.markdown("""
<div class="testimonial-row">
    <div class="testimonial">
        "The AI Travel Planner saved me hours of research! The itinerary was perfectly tailored to my interests."
        <span class="author">– Priya S., Bangalore</span>
    </div>
    <div class="testimonial">
        "I never would have found those hidden gems without this tool. The transportation recommendations were spot on!"
        <span class="author">– Rajiv M., Delhi</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border-top:1px solid #253241; margin-bottom:15px; margin-top:24px;">', unsafe_allow_html=True)
