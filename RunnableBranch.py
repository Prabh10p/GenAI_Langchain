from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# Prompt to analyze sentiment (JSON output)
prompt1 = PromptTemplate(
    template="""
You are a Sentiment Analyzer.
Analyze the sentiment from the text below.
Return only valid JSON in this format:
{{"sentiment": "positive"}}, {{"sentiment": "negative"}}, or {{"sentiment": "neutral"}}

Text: {user_input}
""",
    input_variables=["user_input"]
)

# Prompts to generate response based on sentiment
prompt2 = PromptTemplate(
    template="You are a Sentiment Analyzer. Write a positive comment based on this feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="You are a Sentiment Analyzer. Write a negative comment based on this feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt_neutral = PromptTemplate(
    template="You are a Sentiment Analyzer. The feedback seems neutral. Write a polite acknowledgment:\n{feedback}",
    input_variables=["feedback"]
)

# Pydantic parser including neutral
class Parser(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment extracted from text"
    )

parser1 = PydanticOutputParser(pydantic_object=Parser)
parser2 = StrOutputParser()

# Chains
chain1 = prompt1 | model | parser1

branchchain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser2),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser2),
    (lambda x: x.sentiment == "neutral", prompt_neutral | model | parser2),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

final_chain = chain1 | branchchain

# Streamlit app
st.header("Sentiment Analyser Chatbot")
user_input = st.text_input("Write a comment to analyze")

if st.button("Analyze") and user_input:
    result = final_chain.invoke(user_input)
    st.write(result)
