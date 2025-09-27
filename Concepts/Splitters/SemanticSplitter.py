from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

text ="""
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence.

Machine learning (ML) is a subfield of AI that enables systems to learn from data and improve performance over time.

Computer vision is another area of AI that focuses on enabling machines to interpret and understand visual information from the world.

Robotics combines AI with physical machines to perform tasks autonomously.

Natural language processing (NLP) is an AI field concerned with enabling machines to understand, interpret, and generate human language."""


# ✅ Fix: use model_name instead of model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create SemanticChunker
splitter = SemanticChunker(embeddings)

# Option A: Get string chunks
chunks = splitter.split_text(text)
print("String chunk:", chunks[0])
print("String chunk2:", chunks[1])
