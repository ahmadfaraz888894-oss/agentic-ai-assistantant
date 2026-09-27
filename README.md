# Research Agent — Groq, Gemini, OpenAI and Ollama

## Quick start on Windows

1. Extract this ZIP into a new folder. Keep your old project until this version works.
2. Double-click **setup_windows.bat** once. It creates a separate Python environment and installs the app's packages. Python 3.10 or newer and internet access are required.
3. Double-click **start_windows.bat**. Keep its terminal window open while chatting.
4. In the browser, open **API keys** in the sidebar. Paste a Groq or Gemini key (or both). You may leave OpenAI blank.
5. Select **Auto**, then send a question. The page shows which provider answered and how long it took.

If your browser does not open, visit http://localhost:8501.

For permanent configuration, copy `.env.example` to `.env` and fill the needed values. You can also copy your existing `.env` into this folder and add `GROQ_API_KEY` and `GEMINI_API_KEY`. The setup script never overwrites an existing `.env`. Restart the app after editing it. A `GOOGLE_API_KEY` environment variable is also accepted as a Gemini fallback.

## Provider and key options

| Provider | Key setting | Create key |
| --- | --- | --- |
| Groq | GROQ_API_KEY | https://console.groq.com/keys |
| Gemini | GEMINI_API_KEY | https://aistudio.google.com/apikey |
| OpenAI | OPENAI_API_KEY | https://platform.openai.com/api-keys |
| Local Ollama | No key | Install Ollama and a tool-capable model locally |
| Tavily web search | TAVILY_API_KEY (optional) | https://app.tavily.com/home |
| Weatherstack | WEATHERSTACK_API_KEY (optional) | https://weatherstack.com/signup/free |

Keys pasted into the interface are password fields held in the current session; they are not written into files or chat history. Blank fields fall back to the saved `.env` key. Clear conversation resets chat only. To stop using a saved provider key in Auto, uncheck that provider in Auto settings. To remove a saved key permanently, edit `.env` and restart. Do not share filled `.env` files or publish this local app without access controls.

## How Auto works

Default order: **Groq → Gemini → OpenAI**. Providers without keys are skipped. You can choose which provider goes first and disable any exhausted provider. Ollama is an optional final fallback, off by default because local generation can be slow. Selecting one provider directly uses only that provider.

One provider runs at a time. On a timeout, quota error, connection failure or other technical error, Auto tries the next enabled provider. A successful response, including a refusal, is returned as-is. A fallback may send your question and recent conversation to another enabled provider. API usage follows each provider's plan and may incur charges on paid accounts. Free tiers have limits; additional keys on the same account do not necessarily increase them.

The default time limit is 30 seconds per complete cloud-provider attempt, including tool calls; Ollama gets 90 seconds. A hard async timeout cancels an overdue attempt. SDK retries are disabled. Trying several unavailable providers adds their waiting times. You can adjust these settings, select a provider directly, or disable an exhausted one to avoid repeated waits. Changing limits or providers does not guarantee faster responses.

**Prefer quicker responses** lowers reasoning effort for the default Groq GPT-OSS model and turns thinking off for Gemini 2.5 Flash. Turn it off for harder questions. Maximum output defaults to 2,048 tokens and the agent includes the last six completed chat exchanges. Model names can be edited as provider availability changes.

Search and weather require their separate keys. They can be disabled independently. A failed tool reports missing live data to the model; changing the chat provider will not fix an exhausted Tavily or Weatherstack quota.

## Command-line setup (Windows, macOS or Linux)

```text
python -m venv .venv
```

Activate it:
- Windows Command Prompt: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

```text
python -m pip install -r requirements.txt
python -m streamlit run apps.py
```

Use `python3` if your system uses that name. `requirements-tested.txt` records the complete package versions used for the included checks; `requirements.txt` allows compatible updates.

## Project files

- `apps.py`: chat page, password key inputs, provider/model selection and retry controls.
- `agent_core.py`: provider adapters, live tools, timeout and fallback handling.
- `tests/test_agent.py`: offline graph, timeout, fallback, tool and UI checks.
- `research/agent_demo.ipynb`: original notebook kept for reference. It uses the original older LangChain API and is not the entry point for this updated app.

Run checks from this folder with `python -m unittest discover -s tests -v`. They use mock service responses and models, not your keys. Real account quotas, billing and model responses must be verified on your computer.
