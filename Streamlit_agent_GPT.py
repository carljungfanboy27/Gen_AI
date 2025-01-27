import streamlit as st
import openai
import re

# Set up OpenAI API key
openai.api_key = "your_openai_api_key_here"

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

# Streamlit App
st.title("GPT-3.5 Turbo Chatbot")

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

# Option to clear conversation
if st.button("End Conversation"):
    st.session_state.conversation_memory = []
    st.write("Conversation ended. Feel free to come back anytime!")
