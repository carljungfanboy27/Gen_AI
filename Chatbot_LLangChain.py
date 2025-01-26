
!pip install -q openai
!pip install -q llama-index
!pip install -q llama-index-experimental
!pip install -q pypdf
!pip install -q docx2txt
import os
import openai
#os.environ["OPENAI_API_KEY"] = "<the key>"
openai.api_key = os.environ["OPENAI_API_KEY"]
​
import sys
import shutil
import glob
import logging
from pathlib import Path
​
import warnings
warnings.filterwarnings('ignore')
​
import pandas as pd
​
import llama_index
​
## Llamaindex readers
from llama_index.core import SimpleDirectoryReader
​
## LlamaIndex Index Types
from llama_index.core import ListIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import TreeIndex
from llama_index.core import KeywordTableIndex
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core import DocumentSummaryIndex
from llama_index.core import KnowledgeGraphIndex
from llama_index.experimental.query_engine import PandasQueryEngine
​
​
## LlamaIndex Context Managers
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import Node
​
## LlamaIndex Templates
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
​
## LlamaIndex Callbacks
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import LlamaDebugHandler
​
import logging
​
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
​
#model="gpt-4o"
model="gpt-4o-mini"
​
Settings.llm = OpenAI(temperature=0, 
                      model=model, 
                      #max_tokens=512
                      PRESENCE_PENALTY=-2,
                      TOP_P=1,
                     )
​
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
#Settings.llm = Ollama(model="llama3.2", request_timeout=300.0)
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
CUSTOM_PROMPT = [
    ChatMessage(
        content=(
            "You are a conversational assistant designed to encourage exploration and discovery through thoughtful questions.\n"
            "Start interactions by asking open-ended questions relevant to the user's response, with the aim of understanding their perspective and goals better. Only provide instructions or guidance when explicitly requested or when it becomes clear that offering help will improve the conversation.\n"
            "Always answer the query using only the provided context information, "
            "Some rules to follow:\n"
            "1. Never directly reference the given context in your answer.\n"
            "2. Avoid statements like 'Based on the context, ...' or "
            "'The context information ...' or anything along "
            "those lines."
        ),
        role=MessageRole.SYSTEM,
    ),
    ChatMessage(
        content=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]
​
​
CHAT_PROMPT = ChatPromptTemplate(message_templates=CUSTOM_PROMPT)
chat_engine = vectorstoreindex.as_chat_engine(chat_mode="condense_question",
                                              verbose=True,
                                              text_qa_template=CHAT_PROMPT)
chat_engine.reset()
chat_engine.chat_repl()
from llama_index.core.memory import ChatMemoryBuffer
​
memory = ChatMemoryBuffer.from_defaults(token_limit=80000)
chat_engine = vectorstoreindex.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources"
            "answer the query.\n"
            "If the query is unrelated to the context, just answer: I don't know"
            "Query: {query_str}\n"
            "Answer: "
    ),
)
​
chat_engine.reset()
chat_engine.chat_repl()
​
