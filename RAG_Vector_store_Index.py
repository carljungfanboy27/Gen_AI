!pip install -q openai
!pip install -q llama-index==0.12.0
!pip install -q llama-index-core==0.12.0
!pip install -q llama-index-experimental
!pip install -q llama-index-llms-openai
!pip install -q llama-index-llms-ollama
!pip install -q llama-index-embeddings-huggingface
!pip install -q llama-index-embeddings-instructor
!pip install -q llama-index-postprocessor-colbert-rerank
!pip install -q transformers 
!pip install -q torch
!pip install -q pypdf
!pip install -q docx2txt

import os
import sys
import shutil
import glob
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
​
import openai
import llama_index
​
## LlamaIndex Callbacks
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import LlamaDebugHandler
​
import nest_asyncio 
nest_asyncio.apply()
#nest_asyncio.suppress()
import logging

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import logging
​
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#os.environ["OPENAI_API_KEY"] = "<the key>"
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model="gpt-3.5-turbo"
#model="gpt-4o"
#model="gpt-4o-mini"

Settings.llm = OpenAI(temperature=0, model=model, PRESENCE_PENALTY=-2, TOP_P=1,)
#Settings.llm = Ollama(model="llama3.2:1b", request_timeout=3600.0)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#os.environ["OPENAI_API_KEY"] = "<the key>"
openai.api_key = os.environ["OPENAI_API_KEY"]
​
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
​
model="gpt-3.5-turbo"
#model="gpt-4o"
#model="gpt-4o-mini"
​
Settings.llm = OpenAI(temperature=0, model=model, PRESENCE_PENALTY=-2, TOP_P=1,)
#Settings.llm = Ollama(model="llama3.2:1b", request_timeout=3600.0)
​
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
DOCS_DIR = "C:/Users/nilsp/OneDrive/Dokumente/AI-Tool papers/Procrastination/"
PERSIST_DIR = "C:/Users/nilsp/OneDrive/Dokumente/AI-Tool papers/Procrastination/Index/"

print(f"Current dir: {os.getcwd()}")

if not os.path.exists(DOCS_DIR):
  os.mkdir(DOCS_DIR)
docs = os.listdir(DOCS_DIR)
docs = [d for d in docs]
docs.sort()
print(f"Files in {DOCS_DIR}")
for doc in docs:
    print(doc)
DOCS_DIR = "C:/Users/nilsp/OneDrive/Dokumente/AI-Tool papers/well-being/"
PERSIST_DIR = "C:/Users/nilsp/OneDrive/Dokumente/AI-Tool papers/well-being/Index/"
​
print(f"Current dir: {os.getcwd()}")
​
if not os.path.exists(DOCS_DIR):
  os.mkdir(DOCS_DIR)
docs = os.listdir(DOCS_DIR)
docs = [d for d in docs]
docs.sort()
print(f"Files in {DOCS_DIR}")
for doc in docs:
    print(doc)
from llama_index.core import Document

text_list = [f"{DOCS_DIR}"]
documents = [Document(text=t) for t in text_list]

# build index
#tmp_index = VectorStoreIndex.from_documents(documents)

documents
from llama_index.core import Document
​
text_list = [f"{DOCS_DIR}Procrastination1", f"{DOCS_DIR}Procrastination2",]
documents = [Document(text=t) for t in text_list]
​
# build index
#tmp_index = VectorStoreIndex.from_documents(documents)
​
documents
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=[f"{DOCS_DIR}Procrastination2"]).load_data() 
documents
from llama_index.core import SimpleDirectoryReader
​
documents = SimpleDirectoryReader(input_files=[f"{DOCS_DIR}Procrastination2"]).load_data() 
documents
from llama_index.core.schema import Node
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.core.node_parser import TokenTextSplitter
#from llama_index.core.node_parser import HTMLNodeParser
#from llama_index.core.node_parser import JSONNodeParser
#from llama_index.core.node_parser import MarkdownNodeParser
#from llama_index.core.node_parser import CodeSplitter
#from llama_index.core.node_parser import SemanticSplitterNodeParser
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.core.ingestion import IngestionPipeline

# parse nodes
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

# build index
#tmp_index = VectorStoreIndex(nodes)

print(len(nodes))

for n in nodes:
    print(n)
    break
from llama_index.core.schema import Node
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.core.node_parser import TokenTextSplitter
#from llama_index.core.node_parser import HTMLNodeParser
#from llama_index.core.node_parser import JSONNodeParser
#from llama_index.core.node_parser import MarkdownNodeParser
#from llama_index.core.node_parser import CodeSplitter
#from llama_index.core.node_parser import SemanticSplitterNodeParser
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.core.ingestion import IngestionPipeline
​
# parse nodes
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
​
# build index
#tmp_index = VectorStoreIndex(nodes)
​
print(len(nodes))
​
for n in nodes:
    print(n)
    break
doc_to_index = f"{DOCS_DIR}Procrastination1"
doc_to_index = f"{DOCS_DIR}Procrastination1"
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage

splitter = SentenceSplitter(chunk_size=2048)

def create_retrieve_index(index_path, index_type):
    if not os.path.exists(index_path):
        print(f"Creating Directory {index_path}")
        os.mkdir(index_path)
    if os.listdir(index_path) == []:
        print("Loading Documents...")
        documents = SimpleDirectoryReader(input_files=[doc_to_index]).load_data()
        print("Creating Index...")
        index = index_type.from_documents(documents,
                                          transformations=[splitter],
                                          show_progress=True,
                                          )
        print("Persisting Index...")
        index.storage_context.persist(persist_dir=index_path)
        print("Done!")
    else:
        print("Reading from Index...")
        index = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir=index_path))
        print("Done!")
    return index
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage
​
splitter = SentenceSplitter(chunk_size=2048)
​
def create_retrieve_index(index_path, index_type):
    if not os.path.exists(index_path):
        print(f"Creating Directory {index_path}")
        os.mkdir(index_path)
    if os.listdir(index_path) == []:
        print("Loading Documents...")
        documents = SimpleDirectoryReader(input_files=[doc_to_index]).load_data()
        print("Creating Index...")
        index = index_type.from_documents(documents,
                                          transformations=[splitter],
                                          show_progress=True,
                                          )
        print("Persisting Index...")
        index.storage_context.persist(persist_dir=index_path)
        print("Done!")
    else:
        print("Reading from Index...")
        index = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir=index_path))
        print("Done!")
    return index
from llama_index.core import VectorStoreIndex

VECTORINDEXDIR = PERSIST_DIR + 'VectorStoreIndex_1'
vectorstoreindex = create_retrieve_index(VECTORINDEXDIR, VectorStoreIndex)
from llama_index.core import VectorStoreIndex
​
VECTORINDEXDIR = PERSIST_DIR + 'VectorStoreIndex_1'
vectorstoreindex = create_retrieve_index(VECTORINDEXDIR, VectorStoreIndex)
