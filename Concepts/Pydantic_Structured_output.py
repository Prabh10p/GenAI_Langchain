from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Use a smaller, free model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",  # ✅ this one exists
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

# Define structured schema
class Review(BaseModel):
    themes: list[str] = Field(description="Key themes from the text")
    summary: str = Field(description="Brief summary of the text")
    sentiment: Literal['pos', 'neg'] = Field(description="Overall sentiment of the text")
    pros: Optional[list[str]] = Field(default=None, description="Positive aspects mentioned")
    cons: Optional[list[str]] = Field(default=None, description="Negative aspects mentioned")

# Wrap model with structured output
structured_model = model.with_structured_output(Review)

# Run inference
result = structured_model.invoke(
    """Deep learning made it possible to move beyond the analysis of numerical data, by adding 
    the analysis of images, speech and other complex data types... (your long text here)"""
)

# Print structured result
print("Themes:", result.themes)
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Pros:", result.pros)
print("Cons:", result.cons)
