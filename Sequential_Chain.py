""" Smart Interview Preparation Assistant

Problem: Students often struggle to prepare for technical and behavioral interviews 
because they need tailored practice, structured feedback, and improvement tracking.

Take user’s input job role (e.g., Data Analyst).
Generate role-specific questions (technical + HR).
Let the user answer (voice or text).
AI evaluates the answer (clarity, depth, confidence).
Provide personalized improvement tips and next set of harder questions."""

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from dotenv import load_dotenv
import streamlit as st


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-9b-it",
    task = "conversational"
)

model = ChatHuggingFace(llm=llm)



prompt1 = PromptTemplate(
    template="You are an Intelligent Interview Preprator. Generate {length_input} role-specific questions (technical + HR) on {user_input} role",
    input_variables=["length_input","used_input "]
)


prompt2 = PromptTemplate(
    template="Being a Interview Preprator. Take user answer - {user_answer} and evaluates the answer (clarity, depth, confidence). Also Provide personalized improvement tips and next set of harder questions",
    input_variables=["user_answer"]
)



chain = prompt1 | model | StrOutputParser()
chain1 =prompt2 | model | StrOutputParser()


if "answer_generated" not in st.session_state:
       st.session_state.answer_generated=False
st.header("LLM Powered Chatbot")
user_input = st.text_input("What is the role you preparing for")
length_input = st.selectbox("How many Questions you want to generate",[5,10,15,20,25,30,35,40,45,50])
if st.button("Answer"):
    result = chain.invoke(
        {"user_input": user_input,
         "length_input":length_input})
    st.session_state.answer_generated=True
    st.write(result)


if st.session_state.answer_generated:
      user_answer = st.text_area("Write you answer for Improvement and Feedback")
      if st.button("Feedback"):
            result = chain1.invoke({"user_answer":user_answer})
            st.write(result)      

