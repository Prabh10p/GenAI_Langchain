from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("Files/modified_gf2m_dataset-2.csv")
docs = loader.load()
print(docs)