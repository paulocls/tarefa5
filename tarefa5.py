from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.llms import Ollama



#Dividir o texto em chunks
file_path = "./Internet_coisas.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)

texts = text_splitter.split_documents(pages)


#impressão na tela
print(texts[0])
print(f'Quantidade de chunks: {len(texts)}')


#Gerar embedding e criar o banco vetorial FAISS
db = FAISS.from_documents(texts, OllamaEmbeddings(model="llama3.2:latest"))
query = "O que é Internet das coisas?"
docs = db.similarity_search(query)
print(docs[0].page_content)

docs = db.similarity_search(query, k=5)
for i, doc in enumerate(docs):
    print(f"Chunk {i + 1}: {doc.page_content}")

#Configurar o módulo RetrievalQA
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

#Função pergunta
query = "O que é Internet das coisas?"
response = qa_chain.invoke(query)
print("QA Response:", response)







