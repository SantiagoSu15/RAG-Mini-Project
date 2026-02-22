import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# LangSmith 
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["USER_AGENT"] = "rag-lab/1.0"


# LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile")


# Embeddings — HuggingFace 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Pinecone
INDEX_NAME = "rag-cat-guide"

pc = Pinecone()  

if INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(INDEX_NAME)

pc.create_index(
    name=INDEX_NAME,
    dimension=384,   
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(f"Índice '{INDEX_NAME}' creado.")

index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


# Indexación 
PDF_URL = "https://faada.org/docs/GuiaParaEntenderAlGato.pdf"

loader = PyPDFLoader(PDF_URL)
docs = loader.load()
print(f"PDF cargado. Páginas: {len(docs)}")
print(f"Muestra del contenido:\n{docs[0].page_content[:300]}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"\nDocumento dividido en {len(all_splits)} fragmentos.")

document_ids = vector_store.add_documents(documents=all_splits)
print(f"Fragmentos guardados en Pinecone. IDs (primeros 3): {document_ids[:3]}")



retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto en comportamiento felino.
Responde la pregunta usando únicamente el contexto proporcionado.
Si no sabes la respuesta, di "No lo sé".

Contexto:
{context}

Pregunta: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


questions = [
    "¿Por qué quiero un gato??",
    "¿DÓNDE ENCONTRAR UN GATO?",
    "¿QUÉ HAGO SI SE ESCONDEY NO SALE EN VARIOS DÍAS?",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Pregunta: {q}")
    print(f"Respuesta: {rag_chain.invoke(q)}")