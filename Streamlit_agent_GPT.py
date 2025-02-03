import os
import openai
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
import re

PERSIST_DIR = "../../Index/VectorStoreIndex/"
# Set up OpenAI API key
openai.api_key = "your_openai_api_key_here"


# Title
st.title("💬 Im your psychological assistent, how can I help you today?")


# Sidebar for API Key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please add your OpenAI API key to continue.")
        st.stop()

# Load the vector store index
try:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    vectorstoreindex = load_index_from_storage(storage_context=storage_context)
    chat_engine = vectorstoreindex.as_chat_engine(chat_mode="condense_question", verbose=True)
except Exception as e:
    st.error(f"Failed to load the vector store index: {e}")
    st.stop()

# Function to generate chatbot response
def generate_response(prompt):
    """
    Generate a response using the GPT-3.5 Turbo model via OpenAI API.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an empathetic, thoughtful, and helpful AI chatbot. Always provide actionable insights and keep the user engaged in a natural conversation. If the conversation goes in a darker direction, provide a supportive warning or suggest appropriate help."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Tool for analyzing user behavior
def analyze_user_behavior(user_input):
    """
    Simulate checking user input for patterns suggesting emotional distress or boredom.
    """
    # Simple heuristic for detecting distress or boredom (placeholder for sentiment analysis)
    if re.search(r"(sad|depressed|lonely|anxious|help)", user_input, re.IGNORECASE):
        return "distress"
    elif re.search(r"(bored|nothing to do|idle)", user_input, re.IGNORECASE):
        return "boredom"
    return "neutral"

st.write("Welcome! I am a thoughtful, empathetic chatbot here to assist you. Feel free to ask me anything.")

# Conversation memory
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# User input
user_message = st.text_input("You:", "", key="user_input")
if st.button("Send") and user_message:
    # Problem detection
    detected_state = analyze_user_behavior(user_message)

    # Build the conversational input
    chatbot_input = user_message
    st.session_state.conversation_memory.append(f"User: {user_message}")

    # Generate a response
    response_text = generate_response(chatbot_input)

    # Check for warnings
    warning = None
    if detected_state == "distress":
        warning = "It seems you might be feeling distressed. Please remember that you are not alone, and help is available if needed."
    elif detected_state == "boredom":
        warning = "It seems you might be feeling bored. Let's find something productive or engaging to talk about!"

    # Append response to memory
    st.session_state.conversation_memory.append(f"Chatbot: {response_text}")

    # Display the response
    st.write(f"Chatbot: {response_text}")
    if warning:
        st.warning(warning)

# Display conversation history
if st.session_state.conversation_memory:
    st.write("### Conversation History:")
    for msg in st.session_state.conversation_memory:
        st.text(msg)

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat_engine.chat(st.session_state.messages[-1]["content"])
                st.write(response.response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.response}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")


# Option to clear conversation
if st.button("End Conversation"):
    st.session_state.conversation_memory = []
    st.write("Conversation ended. Feel free to come back anytime!")
