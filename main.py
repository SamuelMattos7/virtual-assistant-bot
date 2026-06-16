import os
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import chromadb
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la página en Streamlit
st.set_page_config(page_title="DeepSeek PDF Chat", page_icon="📚", layout="centered")
st.title("📚 Chat con tus PDFs (DeepSeek)")

# 3. Cargar variables de entorno y clave de API
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# 4. Configuración del directorio de documentos
DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

# 5. Funciones cacheadas para no recargar la BD en cada interacción
@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    # Carga de PDFs
    def load_pdfs_from_directory(directory: Path):
        documents = []
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            st.warning(f"No se encontraron PDFs en './{directory}/'. Agrega archivos PDF y recarga la página.")
            st.stop() # Detiene la ejecución en lugar de sys.exit()
            
        for pdf_path in pdf_files:
            try:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        documents.append({"text": text, "metadata": {"source": pdf_path.name, "page": page_num}})
            except Exception as e:
                st.error(f"Error leyendo {pdf_path.name}: {e}")
                
        return documents

    # Barra de progreso para la inicialización
    progress = st.progress(0, text="Cargando modelo de embeddings...")
    embeddings = load_embeddings()
    progress.progress(33, text="Conectando a ChromaDB...")

    # Configuración de BD Vectorial
    chroma_client = chromadb.EphemeralClient()

    is_new = True
    chroma_client.delete_collection(name="rag_collection") if "rag_collection" in [c.name for c in chroma_client.list_collections()] else None

    if is_new:
        raw_docs = load_pdfs_from_directory(DOCS_DIR)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for doc in raw_docs:
            split_texts = text_splitter.split_text(doc["text"])
            for chunk_text in split_texts:
                chunks.append({"text": chunk_text, "metadata": doc["metadata"]})
                
        if len(chunks) == 0:
            st.error("No se pudo extraer texto de los PDFs.")
            st.stop()
            
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
                flattened_docs = [doc for sublist in results["documents"] for doc in sublist]
                return "\n\n".join(flattened_docs)
            return ""
            
        retrieve_fn = retrieve_new
    else:
        def retrieve_existing(query: str):
            query_embedding = embeddings.embed_query(query)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            if results and results.get("documents") and results["documents"]:
                flattened_docs = [doc for sublist in results["documents"] for doc in sublist]
                return "\n\n".join(flattened_docs)
            return ""

        retrieve_fn = retrieve_existing

    # Configuración de LLM y Cadena RAG
    progress.progress(66, text="Preparando cadena RAG...")
    llm = ChatOpenAI(
        model="deepseek-chat", 
        openai_api_base="https://api.deepseek.com",
        temperature=0,
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

    chain = (
        {"context": retrieve_fn, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    progress.progress(100, text="¡Listo!")
    progress.empty()
    return chain

# 6. UI de subida de archivos
st.subheader("📄 Sube tus PDFs")
uploaded_files = st.file_uploader(
    "Sube uno o más archivos PDF",
    type="pdf",
    accept_multiple_files=True
)

# Opción para mantener archivos entre recargas (simula el botón "Keep")
keep_uploaded = st.checkbox("Mantener archivos subidos entre recargas", value=True)

if uploaded_files:
    for f in uploaded_files:
        with open(DOCS_DIR / f.name, "wb") as out:
            out.write(f.read())
    initialize_rag_system.clear()  # ← forces rebuild with new PDFs
    st.success(f"{len(uploaded_files)} archivo(s) cargado(s) correctamente.")
else:
    saved_pdfs = list(DOCS_DIR.glob("*.pdf"))
    if keep_uploaded and saved_pdfs:
        st.info(f"Usando {len(saved_pdfs)} archivo(s) PDF previamente subido(s).")
    else:
        st.info("Por favor sube al menos un PDF para comenzar.")
        st.stop()  # ← stops here until user uploads something

# 8. Inicializar sistema con un spinner visual
with st.spinner("Conectando con la base de datos y preparando DeepSeek..."):
    rag_chain = initialize_rag_system()

# 4. Gestión del estado del chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Estoy listo. Hazme preguntas sobre tus PDFs."}]

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Interfaz de entrada de chat
if user_input := st.chat_input("Escribe tu pregunta aquí..."):
    # Agregar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                # Guardar respuesta en el estado
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Ocurrió un error al contactar a la API: {e}")