from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.tools import tool,DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import requests
import os 
import streamlit as st
from dotenv import load_dotenv

api_key = os.getenv("WEATHERSTACK_KEY", "6950e3ee5b5ed3332f1ce3aacf43ffbe")
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
search_tool = DuckDuckGoSearchRun()
api_key = "6950e3ee5b5ed3332f1ce3aacf43ffbe"

@tool
def get_weather(city:str) -> str:
   """This function should return the weather for the city"""
   url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
   response = requests.get(url)
   return response.json()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
   llm = model,
   tools = [search_tool,get_weather],
   prompt = prompt
)

executor = AgentExecutor(
   agent=agent,
   tools=[search_tool,get_weather],
   verbose = True
)



st.header("🌦️ AI Weather Agent")
inp = st.text_input("What you want to Know?")

if st.button("Answer"):
    response = executor.invoke({"input": inp})
    st.write(response["output"])

