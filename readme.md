# Introducción a RAGs con LangChain, OpenAI/Groq y Pinecone

Este proyecto demuestra la implementación de un **RAG** utilizando LangChain como framework, Pinecone como base de datos vectorial, y dos variantes: una con OpenAI y otra con Groq + HuggingFace (gratuita).

Siguiendo el tutorial: `https://python.langchain.com/docs/tutorials/rag/`

## Arquitectura

```
Documento (PDF/Web)
        ↓
   Dividir en chunks (RecursiveCharacterTextSplitter)
        ↓
   Generar embeddings (OpenAI / HuggingFace)
        ↓
   Guardar en Pinecone (Vector Store)
        ↓
   Pregunta del usuario
        ↓
   Recuperar chunks relevantes (Retriever)
        ↓
   Generar respuesta con contexto (GPT-4o-mini / LLaMA 3.3 via Groq)
```

## Archivos

| Archivo | Descripción |
|---|---|
| `RAG-openai.py` | RAG usando **OpenAI** para embeddings (`text-embedding-3-large`) y LLM (`gpt-4o-mini`). Indexa un PDF desde un link. |
| `RAG-.py` | RAG usando **HuggingFace** para embeddings (`all-MiniLM-L6-v2`, corre local) y **Groq** como LLM (`llama-3.3-70b-versatile`).. |

Ambos archivos indexan el mismo documento y exponen la misma cadena RAG — la diferencia es únicamente el proveedor de embeddings y LLM.

## Empezando

### Prerequisitos

```
- Python 3.10 o mayor
- Pinecone API Key 
- OpenAI API Key 
- Groq API Key 
- LangSmith API Key 
- pip package manager
```

### Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/SantiagoSu15/RAG-Mini-Project.git
```

2. **Crear un entorno virtual**
```bash
py -m venv venv
```

3. **Activar el entorno virtual**
```bash
venv\Scripts\activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Configurar variables de entorno

**Para la versión OpenAI (`RAG-openai.py`):**
```bash
set OPENAI_API_KEY=your_openai_key
set PINECONE_API_KEY=your_pinecone_key
set LANGSMITH_API_KEY=your_langsmith_key
```

**Para la versión Groq (`RAG-.py`):**
```bash
set GROQ_API_KEY=your_groq_key
set PINECONE_API_KEY=your_pinecone_key
set LANGSMITH_API_KEY=your_langsmith_key
```

### Ejecutar

```bash
# OpenAI
python RAG.py

# Groq + HuggingFace 
python RAG-.py
```

## Actividad

Ambos scripts implementan un RAG completo con los siguientes pasos:

1. **Carga del documento** — `RAG-openai.py` usa `PyPDFLoader` para cargar un PDF desde una URL.
2. **Chunking** — El documento se divide en fragmentos de 1000 caracteres con overlap de 200 usando `RecursiveCharacterTextSplitter`.
3. **Embeddings y almacenamiento** — Los fragmentos se convierten en vectores y se guardan en un índice de Pinecone.
4. **Cadena RAG** — Ante una pregunta, el retriever busca los 4 fragmentos más relevantes en Pinecone y los inyecta como contexto al LLM para generar la respuesta.

----
**Prueba:**
![Prueba](/prueba.png)



## Autor

* **Santiago Suarez**