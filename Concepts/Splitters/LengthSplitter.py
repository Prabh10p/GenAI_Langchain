from langchain.text_splitter import CharacterTextSplitter

text = "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals"

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""   # character-level split
)

docs = splitter.split_text(text)
#docs = splitter.split_documents(docs). ## for document
print(docs[0])
