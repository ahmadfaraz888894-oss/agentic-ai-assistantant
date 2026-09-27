"""Run with: python -m streamlit run apps.py"""
import streamlit as st
from agent_core import (Settings, DEFAULT_MODELS, KEY_NAMES, PROVIDERS, AgentFailure,
                        ask_agent, load_defaults)

st.set_page_config(page_title="Research Agent", page_icon="✦", layout="centered")
st.markdown("""<style>
.block-container{max-width:920px;padding-top:2rem}
h1{letter-spacing:-.04em}.stChatMessage{border-radius:16px}
</style>""", unsafe_allow_html=True)
defaults = load_defaults()
for key, value in {"messages": [], "failed_question": "", "failure": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.title("✦ Research Agent")
    provider = st.selectbox("Provider", ["Auto", *PROVIDERS])
    st.caption("Auto tries enabled providers in order, skipping those without a key.")
    with st.expander("API keys", expanded=False):
        st.caption("Paste keys here for this session, or keep them in the project .env file. "
                   "Blank fields use .env values. Keys entered here are not written to disk.")
        keys = {}
        for name, env in [*KEY_NAMES.items(), ("Tavily search", "TAVILY_API_KEY"),
                           ("Weatherstack", "WEATHERSTACK_API_KEY")]:
            existing = (defaults.get(env) or "").strip()
            entered = st.text_input(f"{name} key", type="password", key=f"key_{env}",
                        placeholder="Saved key available" if existing else "Paste your key")
            keys[env] = entered.strip() or existing
        st.markdown("[Groq keys](https://console.groq.com/keys) · "
                    "[Gemini keys](https://aistudio.google.com/apikey) · "
                    "[OpenAI keys](https://platform.openai.com/api-keys)")
    with st.expander("Auto settings", expanded=provider == "Auto"):
        first = st.selectbox("Try first", ["Groq", "Gemini", "OpenAI"])
        enabled = []
        for name in ["Groq", "Gemini", "OpenAI"]:
            present = bool(keys[KEY_NAMES[name]])
            if st.checkbox(f"Use {name}", value=True, key=f"enabled_{name}"):
                enabled.append(name)
            st.caption("Key added" if present else "No key — skipped in Auto")
        local_fallback = st.checkbox("Use Ollama as final fallback", value=False)
        st.caption("Ollama runs on your computer and may take longer. "
                   "Each enabled cloud provider may use its account's API credits.")
        auto_order = [n for n in [first, *[p for p in ("Groq", "Gemini", "OpenAI") if p != first]]
                      if n in enabled]
        if local_fallback:
            auto_order.append("Ollama")

    with st.expander("Models and speed"):
        models = {p: st.text_input(f"{p} model", value=defaults.get(f"{p.upper()}_MODEL") or model)
                  for p, model in DEFAULT_MODELS.items()}
        cloud_timeout = st.slider("Time limit per cloud provider (seconds)", 10, 120, 30, 5)
        local_timeout = st.slider("Ollama time limit (seconds)", 30, 300, 90, 15)
        max_tokens = st.select_slider("Maximum output tokens", [512, 1024, 2048, 4096], value=2048)
        fast_mode = st.checkbox("Prefer quicker responses", value=True,
            help="Reduces reasoning effort for the default Groq model and disables thinking "
                 "for Gemini 2.5 Flash. Complex questions may benefit from turning this off.")
        ollama_url = st.text_input("Ollama URL", defaults.get("OLLAMA_BASE_URL") or "http://localhost:11434")
    with st.expander("Search and weather"):
        search_enabled = st.checkbox("Enable web search", value=True)
        weather_enabled = st.checkbox("Enable weather", value=True)
        st.caption("Search key: " + ("added" if keys["TAVILY_API_KEY"] else "missing"))
        st.caption("Weather key: " + ("added" if keys["WEATHERSTACK_API_KEY"] else "missing"))
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.failed_question = ""
        st.session_state.failure = ""
        st.rerun()

settings = Settings(keys=keys, models=models, auto_order=tuple(auto_order),
    cloud_timeout=cloud_timeout, local_timeout=local_timeout, max_tokens=max_tokens,
    ollama_url=ollama_url.strip(), search_enabled=search_enabled,
    weather_enabled=weather_enabled, fast_mode=fast_mode)
st.title("Your research, one conversation.")
st.caption("Chat with Groq, Gemini, OpenAI, or local Ollama.")
if provider == "Auto":
    active = [p for p in auto_order if p == "Ollama" or keys.get(KEY_NAMES[p])]
    st.caption("Auto order: " + (" → ".join(active) if active else "Add an API key to begin"))
if not st.session_state.messages and not st.session_state.failed_question:
    st.info("Open API keys in the sidebar to add Groq or Gemini. "
            "Then ask a question below. Search and weather keys are optional.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "provider" in message:
            st.caption(f"{message['provider']} · {message['model']} · {message['seconds']} s")
            if message.get("attempts"):
                with st.expander("Provider attempts"):
                    for attempt in message["attempts"]:
                        st.write(f"{attempt['provider']}: {attempt['detail']}")

retry = False
if st.session_state.failed_question:
    with st.chat_message("user"):
        st.markdown(st.session_state.failed_question)
    st.error(st.session_state.failure)
    retry = st.button("Retry with current settings")
question = st.chat_input("Ask a question…", max_chars=12000)
if retry:
    question = st.session_state.failed_question
if question and question.strip():
    question = question.strip()
    st.session_state.failed_question = ""
    st.session_state.failure = ""
    if not retry:
        with st.chat_message("user"):
            st.markdown(question)
    with st.status("Connecting…", expanded=True) as status:
        def show_status(text):
            status.update(label=text)
            status.write(text)
        try:
            result = ask_agent(st.session_state.messages, question, provider, settings, show_status)
        except (AgentFailure, ValueError) as error:
            st.session_state.failed_question = question
            st.session_state.failure = str(error)
            status.update(label="Could not complete this request", state="error")
        except Exception:
            st.session_state.failed_question = question
            st.session_state.failure = "Unexpected error. Check your installation and try again."
            status.update(label="Could not complete this request", state="error")
        else:
            st.session_state.messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.pop("answer"), **result}])
            status.update(label="Complete", state="complete", expanded=False)
    st.rerun()
