from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    lang="en"
)

query = "History of Machine Learning"

docs = retriever.get_relevant_documents(query)


for i,doc in enumerate(docs):
      print(f"\n=== Document {i} ===")
      print(doc.page_content[:500])  