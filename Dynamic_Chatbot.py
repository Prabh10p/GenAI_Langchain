from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Use a smaller, free model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it", 
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

st.header("Chatbot Using LangChain")

# User inputs
user_input = st.text_input("What do you want to ask?")
topic_input = st.selectbox(
    "What topic you want to know about",
    ['AI', 'ML', 'DL', 'NLP', 'GEN AI', 'LangChain']
)
length_input = st.selectbox(
    "What length you want",
    ["small", "medium", "large"]
)

# Define template
prompt = PromptTemplate(
    template="You are a Smart Assistant. Based on the question: {user_input}, "
             "give me a {length_input} answer about {topic_input}.",
    input_variables=["user_input", "topic_input", "length_input"]
)

# Button to trigger answer
if st.button("Answer"):
    # Fill the template
    final_prompt = prompt.format(
        user_input=user_input,
        topic_input=topic_input,
        length_input=length_input
    )
    
    # Get model response
    result = model.invoke(final_prompt)
    st.write(result.content)
