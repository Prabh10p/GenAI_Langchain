
""" Problem Statement: Multi-Aspect Product Review Analyzer

Scenario:
E-commerce platforms receive thousands of product reviews daily. Each review contains valuable information, but it’s often unstructured and covers different aspects like quality, delivery, customer service, and value for money.
Objective:
Build an AI system that analyzes a product review in parallel for multiple dimensions and provides a structured summary.
Requirements:
Input: A single product review text (e.g., 2–3 sentences).



Parallel Chains:
Chain 1 → Analyze product quality sentiment and key points.
Chain 2 → Analyze delivery experience sentiment and key points.
Chain 3 → Analyze customer service experience sentiment and key points.
Chain 4 → Analyze overall value for money sentiment and key points.


Output:
Combine the results from all chains into a structured report:
Bonus:
Generate an overall recommendation (buy/not buy) based on combined sentiments.
Include key improvement suggestions from negative aspects."""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import streamlit as st

# Load API keys
load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# Define schema for aspect analysis
class AspectAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the review for this aspect"
    )
    summary: str = Field(
        description="Brief summary of key points related to this aspect"
    )

parser = PydanticOutputParser(pydantic_object=AspectAnalysis)

# Add format instructions
format_instructions = parser.get_format_instructions()

prompt1 = PromptTemplate(
    template=(
        "Analyze ONLY the product quality in this review:\n"
        "{user_input}\n\n"
        "Provide the output in **strict JSON** format.\n"
        "{format_instructions}"
    ),
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
)

prompt2 = PromptTemplate(
    template=(
        "Analyze ONLY the delivery experience in this review:\n"
        "{user_input}\n\n"
        "Provide the output in **strict JSON** format.\n"
        "{format_instructions}"
    ),
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
)

prompt3 = PromptTemplate(
    template=(
        "Analyze ONLY the customer service in this review:\n"
        "{user_input}\n\n"
        "Provide the output in **strict JSON** format.\n"
        "{format_instructions}"
    ),
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
)

prompt4 = PromptTemplate(
    template=(
        "Analyze ONLY the value for money in this review:\n"
        "{user_input}\n\n"
        "Provide the output in **strict JSON** format.\n"
        "{format_instructions}"
    ),
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
)


# Run in parallel
parallel_chain = RunnableParallel({
    "product_quality": prompt1 | model | parser,
    "delivery": prompt2 | model | parser,
    "customer_service": prompt3 | model | parser,
    "value_for_money": prompt4 | model | parser,
})

# Overall recommendation summarizer
summary_prompt = PromptTemplate(
    template=(
        "You are an AI Review Summarizer.\n\n"
        "Given the following aspect-level analyses:\n"
        "Product Quality: {product_quality}\n"
        "Delivery: {delivery}\n"
        "Customer Service: {customer_service}\n"
        "Value for Money: {value_for_money}\n\n"
        "Task:\n"
        "- Generate an overall recommendation: 'Buy' or 'Not Buy'\n"
        "- Highlight key strengths (positive aspects)\n"
        "- Suggest improvements for negative aspects\n"
        "- Keep the response structured and concise."
    ),
    input_variables=["product_quality", "delivery", "customer_service", "value_for_money"]
)





sequential_chain =  summary_prompt | model | StrOutputParser()


final_chain = parallel_chain | sequential_chain

result = final_chain.invoke("The product feels cheap and broke after two days. Delivery was late, and the customer service was unhelpful when I asked for a replacement. Definitely not worth the money")
print(result)