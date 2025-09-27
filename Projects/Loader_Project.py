# Make a application that take multiple thing like  
# text or folder or Pdf to read from it and then generating a Summary and Quiz n that for practise

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader,TextLoader, DirectoryLoader, WebBaseLoader
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()



prompt1 = PromptTemplate(
    template = "Summarize the following text deeply /n {input_text}",
    input_variables=["input_text"]
)



prompt2 =PromptTemplate(
    template = "Generate best learnable {text_inp} MCQ Quiz questions from following text/n {input_text}",
    input_variables=["input_text"]
)



chain1 = prompt1 |model| parser

chain2 = prompt2|model|parser



st.header("Summariser and Quiz Generator Macchine")


input_type = st.radio(
    "Choose your input type:",
    ("Text", "PDF", "Folder")
)

if input_type =="Text":
    text = st.text_area("Which text you want you Summarise?")
    loader = TextLoader(text)


if input_type =="PDF":
    text = st.file_uploader("Upload you Pdf to Summarise?",type=["pdf"])
    loader = PyPDFLoader(text)


if input_type =="Folder":
    text = st.file_uploader("Upload your Folder you want to Summarise?")
    loader = DirectoryLoader(text)


if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False


if st.button("Summarize"):
    result = chain1.invoke({'input_text':text})
    st.subheader("Summary")
    st.write(result)
    st.session_state.summary_generated=True


if  st.session_state.summary_generated:
    text_inp = st.selectbox("How many MCQS you want to generate?",[5,10,15,20,25,30,35,40,50])
    if st.button("Generate Quiz"):
        result = chain2.invoke({'text_inp':text_inp,'input_text':text})
        st.subheader("Quiz")
        st.write(result)