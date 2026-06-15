import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "apikey"  # Pon tu clave de DeepSeek aquí

if "sk-xxx" in os.environ["OPENAI_API_KEY"]:
    print("Error: Por favor, reemplaza el texto de marcador con tu clave API real de DeepSeek.")
    sys.exit(1)

DOCS_DIR = Path("documents")
DB_DIR = "chroma_db"
DOCS_DIR.mkdir(exist_ok=True)

# 2. Carga de PDFs usando PyMuPDF
def load_pdfs_from_directory(directory: Path):
    documents = []
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No se encontraron PDFs en './{directory}/'. Agrega archivos PDF y reinicia.")
        sys.exit(0)
        
    print(f"Cargando {len(pdf_files)} archivo(s) PDF...")
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    documents.append({"text": text, "metadata": {"source": pdf_path.name, "page": page_num}})
        except Exception as e:
            print(f"Error leyendo {pdf_path.name}: {e}")
            
    return documents

# 3. Configuración Inteligente de la Base de Datos Vectorial
def get_retriever():
    # Inicializamos el modelo de embeddings local
    print("Inicializando modelo de embeddings local...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Conectamos con el cliente persistente de Chroma
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    # Intentamos obtener la colección existente para ver si ya tiene datos
    try:
        collection = chroma_client.get_collection(name="rag_collection")
        if collection.count() > 0:
            print(f"Base de datos detectada en '{DB_DIR}' con {collection.count()} fragmentos. Cargando directamente...")
            
            # Definimos la función de recuperación usando los datos guardados
            def retrieve_existing(query: str):
                query_embedding = embeddings.embed_query(query)
                results = collection.query(query_embeddings=[query_embedding], n_results=3)
                if results and results.get("documents") and results["documents"]:
                    # CORRECCIÓN: Aplanamos la lista de listas que devuelve ChromaDB
                    flattened_docs = [doc for sublist in results["documents"] for doc in sublist]
                    return "\n\n".join(flattened_docs)
                return ""
            return retrieve_existing
    except Exception:
        # Si la colección no existe, continuamos al proceso de creación normal
        pass

    # PROCESO DE INDEXACIÓN INICIAL (Solo si la base de datos está vacía)
    print("Base de datos no encontrada o vacía. Iniciando indexación de documentos...")
    raw_docs = load_pdfs_from_directory(DOCS_DIR)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in raw_docs:
        split_texts = text_splitter.split_text(doc["text"])
        for chunk_text in split_texts:
            chunks.append({"text": chunk_text, "metadata": doc["metadata"]})
            
    print(f"Creados {len(chunks)} fragmentos (chunks).")
    
    if len(chunks) == 0:
        print("\n[!] ERROR: No se pudo extraer texto de los PDFs.")
        sys.exit(1)
        
    print(f"Generando embeddings locales y guardando en {DB_DIR}...")
    collection = chroma_client.get_or_create_collection(name="rag_collection")
    
    texts_to_embed = [c["text"] for c in chunks]
    embedded_docs = embeddings.embed_documents(texts_to_embed)
    
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embedded_docs,
        documents=texts_to_embed,
        metadatas=[c["metadata"] for c in chunks]
    )
    
    def retrieve_new(query: str):
        query_embedding = embeddings.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        if results and results.get("documents") and results["documents"]:
            # CORRECCIÓN: Aplanamos la lista de listas que devuelve ChromaDB
            flattened_docs = [doc for sublist in results["documents"] for doc in sublist]
            return "\n\n".join(flattened_docs)
        return ""
        
    return retrieve_new

retrieve_fn = get_retriever()

# Configuración corregida para evitar bloqueos de CloudFront
llm = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_base="https://api.deepseek.com",  # URL limpia sin '/v1'
    temperature=0,
    # Añadimos cabeceras adicionales si CloudFront se pone estricto con los scripts
    default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)

system_prompt = (
    "Eres un asistente para tareas de respuesta a preguntas. "
    "Utiliza los siguientes fragmentos de contexto recuperados para responder "
    "la pregunta. Si no sabes la respuesta, di que no la sabes y responde en español.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = (
    {"context": retrieve_fn, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Interfaz de Terminal
print("\n¡Chatbot listo (Conectado a DeepSeek)! Escribe 'exit' o 'quit' para salir.\n")
while True:
    try:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            print("¡Adiós!")
            break
            
        print("DeepSeek está pensando...")
        response = rag_chain.invoke(user_input)
        print(f"\nAsistente: {response}\n")
        
    except KeyboardInterrupt:
        print("\n¡Adiós!")
        break
