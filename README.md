# Agentic Chatbot using Langraph

## Overview

This project is a Streamlit-powered web app that builds an agentic chatbot using Langraph and LangChain tools. It combines:

- a conversational AI frontend in `app.py`
- a tool-enabled backend in `backend.py`
- PDF-based retrieval (RAG) via FAISS vector search
- real-time weather lookup, stock quote retrieval, calculator support, and web search
- human-in-the-loop approval for simulated stock purchases
- Docker support for container deployment

The app stores chat state in `chatbot.db` and document embeddings in `faiss_db`.

## Features

- Chat with an AI assistant using Gemini and LangGraph
- Upload a PDF to add document content for Retrieval Augmented Generation (RAG)
- Use a calculator tool for math queries
- Get stock prices via Alpha Vantage
- Request current weather via OpenWeatherMap
- Perform internet searches using Tavily
- Simulate stock purchase requests with human approval
- Store and switch between chat threads

## Files and purpose

- `app.py`
  - Streamlit application and frontend interface
  - Session-state handling for chat threads, pending human approvals, and message history
  - PDF upload handling and user input streaming
  - UI styling and sidebar conversation navigation

- `backend.py`
  - Defines the LLM and tool chain
  - Loads environment variables with `dotenv`
  - Implements PDF ingestion, retrieval, and tool functions
  - Configures the LangGraph `StateGraph` and SQLite checkpoint
  - Exposes helper functions for the frontend

- `requirements.txt`
  - Python dependencies required by the project
  - Includes LangGraph, Streamlit, LangChain, FAISS, PyPDF, and API clients

- `Dockerfile`
  - Builds a slim Python container image
  - Installs dependencies and exposes Streamlit on port `8501`
  - Runs `streamlit run app.py` in headless mode

## Environment variables

Create a `.env` file in the project root with at least these values:

```bash
GOOGLE_API_KEY=your_google_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key
```

> Note: `backend.py` uses `load_dotenv()`, so these values are loaded automatically when the app starts.

## Setup and installation

### 1. Clone or open the repository

Open the project folder in VS Code or clone it locally.

### 2. Create a Python virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create a `.env` file

Add the required API keys as shown above.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the browser URL shown in the terminal, usually `http://localhost:8501`.

## How the app works

### Frontend (`app.py`)

1. `st.set_page_config(...)` sets the page title, icon, layout, and sidebar state.
2. Custom CSS is injected with `st.markdown(..., unsafe_allow_html=True)` to style the page, chat bubbles, sidebar, buttons, and form.
3. Helper functions manage thread IDs, chat history, and interrupt state.
4. Session state keys are initialized:
   - `message_history`
   - `thread_id`
   - `chat_threads`
   - `pending_hitl`
5. The sidebar displays `New Chat` and existing chat threads.
6. Chat history is rendered from `st.session_state["message_history"]`.
7. If a stock purchase needs approval, the app displays approval buttons.
8. The chat input accepts text and PDF uploads.
9. When a PDF is uploaded, the file is saved temporarily and passed to `ingest_rag_document(...)`.
10. When the user submits text, the assistant uses `chatbot.stream(...)` to generate a response and optionally call tools.

### Backend (`backend.py`)

1. The file imports LangGraph, LangChain, tool decorators, and document processing utilities.
2. `load_dotenv()` loads environment variables from `.env`.
3. The main LLM is configured using `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`.
4. Embeddings are created with `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`.
5. `ingest_rag_document(file_path)`:
   - loads the PDF with `PyPDFLoader`
   - splits text into chunks with `RecursiveCharacterTextSplitter`
   - either creates or appends to a FAISS index stored in `faiss_db`
6. `get_retriever()` returns a FAISS retriever if the index exists.
7. `rag_tool(query)` is a tool that searches uploaded PDF content and returns matching passages.
8. Additional tools:
   - `calculator(expression)` evaluates math expressions safely
   - `get_stock_price(symbol)` fetches stock quotes from Alpha Vantage
   - `purchase_stock(symbol, quantity)` simulates a purchase and uses `interrupt(...)` for human approval
   - `get_current_weather(location)` fetches weather from OpenWeatherMap
9. The tools are combined into `llm_with_tools = llm.bind_tools(tools)`.
10. `chat_node(state)` constructs a system prompt instructing the assistant when to use each tool.
11. The LangGraph state graph assembles the chat node and tool node with checkpoint persistence in `chatbot.db`.
12. `get_all_threads()` reads saved checkpoint configs to populate chat thread history.

## Running with Docker

### Build the image

```bash
docker build -t agentic-chatbot .
```

### Run the container

```bash
docker run --rm -p 8501:8501 -v %cd%:/app agentic-chatbot
```

If you use Docker on Windows, ensure your `.env` file is available inside the container or pass environment variables directly.

## Usage guide

### Start a new conversation

- Click `➕ New Chat` in the sidebar to begin a fresh thread.
- Each thread is saved and shown in the sidebar for later retrieval.

### Ask questions

- Type questions into the chat input at the bottom.
- The assistant can answer plain questions directly and call tools when needed.

### Upload a PDF for RAG

- Attach a PDF by clicking the paperclip icon in the chat input.
- The app processes the PDF and stores its text embeddings.
- Follow-up questions about the PDF will use the uploaded document for answers.

### Use the tools

- Weather questions trigger `get_current_weather`
- Stock quote requests trigger `get_stock_price`
- Math queries use `calculator`
- Document questions use `rag_tool`
- Search queries use `search_tool`
- Stock purchase requests require human approval via the HITL interface

### Human-in-the-loop approval

- If the assistant wants to purchase stock, it stops and waits for your approval.
- Click `✅ Approve Purchase` or `❌ Reject Purchase` to continue.

## Notes and troubleshooting

- `chatbot.db` stores conversation threads and tool state.
- `faiss_db` stores document embeddings for PDF retrieval.
- Remove those folders/files if you want to reset state.
- If a tool returns an error, check that the corresponding API key is correctly set in `.env`.

## Optional improvements

- Add more tools and tool prompts in `backend.py`
- Support more file formats beyond PDF
- Add authentication for the Streamlit app
- Improve thread names with user-friendly labels instead of UUID prefixes

## License

This project is provided under the current repository license.
