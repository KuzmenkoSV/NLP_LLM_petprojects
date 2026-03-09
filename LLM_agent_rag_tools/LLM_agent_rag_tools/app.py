# ================================
# Imports
# ================================

import os
# Пути к моделям
MODEL_DIR = '/mnt/Data/cache/'
os.environ["LANGFUSE_PUBLIC_KEY"] = ...
os.environ["LANGFUSE_SECRET_KEY"] = ...
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

LANGFUSE_BASE_URL="https://cloud.langfuse.com"
MODEL_PATH_BGE = os.path.join(MODEL_DIR, 'bge-m3')

import streamlit as st

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, List

# Vector DB
from langchain_chroma import Chroma
#from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
# Tool decorator
from langchain.tools import tool

# ================================
# LLM CONNECTION
# ================================
from langfuse.callback import CallbackHandler
from langfuse import Langfuse

langfuse = Langfuse()
langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",  # vLLM doesn't require real key
    model="qwen",
    temperature=0.2,
    callbacks=[langfuse_handler] 
)

# ================================
# VECTOR DATABASE
# ================================
# Инициализация модели embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_PATH_BGE,
    model_kwargs={'device': 'cpu'}
)

vector_db = Chroma(
    collection_name="documents",
    embedding_function=embedding_model
)

# ================================
# DOCUMENT INGESTION
# ================================
uploaded_file_name = None

def ingest_file(text: str, filename):
    """
    Takes raw text and stores it into vector database
    """
    global uploaded_file_name
    uploaded_file_name = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.create_documents([text])

    vector_db.add_documents(docs)

# ================================
# TOOL FOR AGENT
# ================================


def create_agent():
    @tool
    def search_documents(query: str) -> str:
        """
        Ищем информацию в документе!
        """
        docs = vector_db.similarity_search(query, k=3)
    
        if not docs:
            return "Нет релевантной информации в документах!"

        context = "\n\n".join([d.page_content for d in docs])
        langfuse.trace(
            name="retrieval",
            input=query,
            output=context
        )
    
        return context
    
    base_description = search_documents.__doc__
    if uploaded_file_name:
        tool_description = f"""
        Ищи информацию в загруженном документе, если уместно: "{uploaded_file_name}".
        Отвечай только на Русском языке!
        Если вопрос является специфическим и возможно связан с загруженным файлов, используй этот инструмент.
        Также используй этот инструмент, если в вопросе есть фразы по типу:
            - "посмотри в файле"
            - "что там в документе"
            - "найди в загруженном файле"
            - "посмотри в отчете"
            - "согласно документу"
            - "на основе файла/из файла скажи"
            - "в загруженном документе написано"
            - "по данным из файла"
            - "что говорится в файле о"
            - "найди информацию из документа"
        """
    else:
        tool_description = base_description + """
        Note: Файл не загружен, использовать инстурмент бессмысленно, он ничего не вернет.
        """
    
    search_documents.__doc__ = tool_description
    # ================================
    # СREATE AGENT
    # ================================
    
    tools = [search_documents]
    llm_with_tools = llm.bind_tools(tools)
    # --- ГРАФ ---
    
    class ChatState(TypedDict):
        messages: List[BaseMessage]
        
    def call_model(state: ChatState):
        """Узел агента"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response]}
    
    
    workflow = StateGraph(ChatState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    #memory = MemorySaver()
    
    app = workflow.compile()
    return app

app = create_agent()
# ================================
# STREAMLIT UI
# ================================

st.title("🧠 Smart Chat Agent (with optional file knowledge)")

st.write(
"""
Агент может
• поддерживать диалог  
• использовать загруженный документ для ответа, если нужно
"""
)

# ================================
# SESSION STATE (MEMORY)
# ================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================================
# FILE UPLOAD
# ================================


uploaded_file = st.file_uploader(
    "Upload a text file (optional)",
    type=["txt"]
)

if uploaded_file:
    text = uploaded_file.read().decode()
    
    # Pass filename to ingest_file
    ingest_file(text, uploaded_file.name)
    
    # Recreate agent with updated filename info
    app = create_agent()
    
    st.success(f"File '{uploaded_file.name}' uploaded and indexed!")

# Показываем, что файл подгрузился
if uploaded_file_name:
    st.info(f"📄 Active document: {uploaded_file_name}")

# ================================
# CHAT INPUT
# ================================

user_input = st.chat_input("Ask something...")

if user_input:
    # Convert session messages to LangChain message format
    langchain_messages = []
    langchain_messages.append(SystemMessage(content="""Ты русскоязычный ассистент. ВАЖНО: Всегда отвечай ТОЛЬКО на русском языке, даже если вопрос содержит английские слова.

Правила:
1. ОТВЕЧАЙ ТОЛЬКО ПО-РУССКИ
2. Используй информацию из загруженного документа, когда это уместно
3. Если в документе нет информации, так и скажи по-русски
4. Будь вежливым и полезным"""))
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    # Add new user message
    langchain_messages.append(HumanMessage(content=user_input))
    
    # Save user message to session state
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    
    result = app.invoke(
        {"messages": langchain_messages},
        config={"callbacks": [langfuse_handler]}
    )
    
    # Extract assistant response
    answer = result["messages"][-1].content
    
    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
# ================================
# DISPLAY CHAT HISTORY
# ================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])