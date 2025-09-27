""" Smart Interview Preparation Assistant

Problem: Students often struggle to prepare for technical and behavioral interviews 
because they need tailored practice, structured feedback, and improvement tracking.

Take user’s input job role (e.g., Data Analyst).
Generate role-specific questions (technical + HR).
Let the user answer (voice or text).
AI evaluates the answer (clarity, depth, confidence).
Provide personalized improvement tips and next set of harder questions."""



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Switch to a smaller, free model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = "You are and Inrterview Expert. Generate {no_of_questions} role-specific questions (technical + HR) based on {user_role} role",
    input_variables=['no_of_questions','user_role']
)


prompt2 = PromptTemplate(
    template = "You are and Interview Expert. Evaluates the (clarity, depth, confidence) of {user_answer} and Provide personalized improvement tips and next set of harder questions",
    input_variables=['user_answer']
)


chain1 = RunnableSequence(prompt1,model,parser)

chain2 = RunnableSequence(prompt2,model,parser)



if 'generated_questions' not in st.session_state:
       st.session_state.generated_questions = False
st.header("Interview Preparation AI")
no_of_questions = st.selectbox("Select No of Questions you want to Generate",[5,10,15,20,25,30,35,40,45,50])
user_role = st.text_input("Type the Role you are Interested in")
if st.button("Answer"):
     result = chain1.invoke({
          'no_of_questions':no_of_questions,
          'user_role':user_role 
     })
     st.session_state.generated_questions=True
     st.write(result)

    
if st.session_state.generated_questions:
     feedback = st.text_area("Provide your answers for Feeback and Improvement")
     if st.button("Feedback"):
        result1 = chain2.invoke({
               'user_answer':feedback}
        )
        st.write(result1)
