from langchain_community.retrievers import ArxivRetriever

# Initialize retriever
retriever = ArxivRetriever(
    load_max_docs=2,          # fetch up to 2 results
    get_full_documents=True   # pull full PDFs instead of just abstracts
)

# Query arXiv
query = "history of machine learning"
docs = retriever.invoke(query)

# Print documents
for i, doc in enumerate(docs, start=1):
    print(f"==Document{i}==")
    print(doc.page_content[:1000])  # limit to 1000 chars for readability
    print()
