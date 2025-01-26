from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

# Load and process PDFs
def load_and_process_pdfs(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    return documents

# Split documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Create FAISS knowledge base
def create_knowledge_base(documents, embedding_model="text-embedding-ada-002"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

pdf_folder = "path_to_pdfs"
documents = load_and_process_pdfs(pdf_folder)
chunks = split_documents(documents)
knowledge_base = create_knowledge_base(chunks)

meta_prompt = """
You are a conversational assistant specializing in psychological well-being. 
Your goal is to help the user think self-reflectively and provide evidence-based knowledge when needed.

Guidelines:
1. Start by asking an open-ended question about the user's thoughts or feelings.
2. Respond with follow-up questions that encourage self-reflection.
3. Use the knowledge base to provide evidence-based insights when the user seeks guidance or information.
4. Transition to providing instructions only when explicitly requested or necessary.

Example:
User: "I'm feeling a bit overwhelmed lately."
Assistant: "What do you think is contributing to that feeling? Could it be related to work, relationships, or something else?"
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=knowledge_base.as_retriever()
)

# Chatbot loop
def chatbot():
    print("Welcome to the Psychological Well-being Chatbot!")
    print("Feel free to share your thoughts or ask questions.\n")
    
    context = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Take care of yourself.")
            break
        
        # Use the meta-prompt for context
        response = qa_chain.run({"question": user_input, "chat_history": context})
        context.append({"role": "user", "content": user_input})
        context.append({"role": "assistant", "content": response})
        
        print(f"Assistant: {response}")

# Run the chatbot
chatbot()
