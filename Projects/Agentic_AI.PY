from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.tools import tool,DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather(city:str) -> str:
   """This function should return the weather for the city"""
   url = 
   response = requests.get(url)
   return response.json()