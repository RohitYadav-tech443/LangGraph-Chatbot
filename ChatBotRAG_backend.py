from __future__ import annotations

from logging import config
import os
import sqlite3
import tempfile
import requests
from typing import Annotated, Any, Dict, Optional, TypedDict    
from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0
)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
embeddings=None
def get_embeddings():
    global embeddings

    if embeddings is None:
        embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )


    return embeddings

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
CURRENT_THREAD_ID = None


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, get_embeddings())
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return str(result)
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    A tool to fetch the current stock price for a given symbol.Using the finnhub.io API.

    Args:
        symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
        dict: A dictionary containing the stock price or an error message.
    """
    try:
        # Using a free API to get stock price
        url=f"https://finnhub.io/api/v1/quote?symbol={symbol}&token=d96kd59r01qr77dkrf9gd96kd59r01qr77dkrfa0"
        response = requests.get(url)
        data = response.json()
        
        if "c" in data:
            return f"The current price of {symbol} is {data['c']}"
        else:
            return {"error": "Could not retrieve stock price. Please check the symbol."}
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str) -> str:
    """
    Search the uploaded PDF and return the relevant context.
    """

    global CURRENT_THREAD_ID

    retriever = _get_retriever(CURRENT_THREAD_ID)

    if retriever is None:
        return "No PDF has been uploaded for this chat."

    docs = retriever.invoke(query)

    return "\n\n".join(doc.page_content for doc in docs)


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    global CURRENT_THREAD_ID

    CURRENT_THREAD_ID = None

    if config and isinstance(config, dict):
        CURRENT_THREAD_ID = config.get(
            "configurable", {}
        ).get("thread_id")

        system_message = SystemMessage(
        content="""
        You are a helpful AI assistant.

        Use rag_tool whenever the user asks anything about the uploaded PDF.

        Use calculator for arithmetic.

        Use get_stock_price for stock prices.

        Use DuckDuckGo search only when external information is required.

        Never invent facts if the tool can answer.
        """
        )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})