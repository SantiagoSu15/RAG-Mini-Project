import os

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["USER_AGENT"] = "rag-lab/1.0"

# LangSmith 
os.environ["LANGSMITH_TRACING"] = "true"


# OpenAI
llm = init_chat_model("gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Pinecone 
INDEX_NAME = "rag-lab-cats"

pc = Pinecone()  

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,          
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


# Indexación — cargar PDF y dividir

PDF_URL = "https://faada.org/docs/GuiaParaEntenderAlGato.pdf"
loader = PyPDFLoader(PDF_URL)
docs = loader.load()

print(f"Páginas cargadas: {len(docs)}")
print(f"Primeros 500 caracteres:\n{docs[0].page_content[:500]}")

# Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"\nDocumento dividido en {len(all_splits)} fragmentos.")

# Guardar en Pinecone
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Documentos guardados. IDs (primeros 3): {document_ids[:3]}")


# RAG Chain

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Prueba
questions = [
    "¿Por qué quiero un gato??",
    "¿DÓNDE ENCONTRAR UN GATO?",
    "¿QUÉ HAGO SI SE ESCONDEY NO SALE EN VARIOS DÍAS?",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Pregunta: {q}")
    print(f"Respuesta: {rag_chain.invoke(q)}")