from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Prepare documents
documents = [
    Document(page_content="Artificial Intelligence is the simulation of human intelligence by machines.", metadata={"id": 1, "topic": "AI"}),
    Document(page_content="Machine Learning is a subfield of AI focused on learning from data.", metadata={"id": 2, "topic": "ML"}),
    Document(page_content="Deep Learning uses neural networks with many layers to model complex patterns.", metadata={"id": 3, "topic": "DL"}),
    Document(page_content="Natural Language Processing (NLP) enables machines to understand human language.", metadata={"id": 4, "topic": "NLP"}),
    Document(page_content="Computer Vision focuses on interpreting and processing visual information.", metadata={"id": 5, "topic": "CV"}),
    Document(page_content="Reinforcement Learning trains agents to make decisions via rewards and penalties.", metadata={"id": 6, "topic": "RL"}),
    Document(page_content="Robotics combines AI with mechanical systems to perform tasks autonomously.", metadata={"id": 7, "topic": "Robotics"}),
]

# Step 2: Define embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vectorstore
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="my_collection"
)

# Step 4: Convert to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # return 3 documents

# Step 5: Run query
query = "History of Machine Learning"
docs = retriever.get_relevant_documents(query)

# Step 6: Print results
for i, doc in enumerate(docs):
    print(f"\n=== Document {i} ===")
    print(doc.page_content[:500])
