from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Files/resume_prabhjot_singh_0002.pdf")
docs = loader.load()
print(docs)
