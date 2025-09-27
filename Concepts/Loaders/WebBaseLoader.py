
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-9b-it",
    task = "coversational"
)



model = ChatHuggingFace(llm=llm)

st.header("WebPage Summariser")
url = "https://en.wikipedia.org/wiki/History_of_artificial_intelligence"
loader = WebBaseLoader(url)
docs = loader.load()
url_input = st.text_input("Provide a link for webpage you want to summarise", url)




prompt = PromptTemplate(
    template = "Summarise the whole article thew article is {user_input}",
    input_variables=["user_input"]
)

if st.button('Summarise'):
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({'user_input':url_input})

    st.write(result)
