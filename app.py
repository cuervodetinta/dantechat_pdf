import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Personalización de estilo en la app
st.markdown("""
    <style>
    body {
        background-color: #FFF8DC !important;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    .stTextInput, .stTextArea, .stNumberInput, .stSlider, .stFileUploader, .stMarkdown, .stButton {
        background-color: #FFFFFF !important;
        color: black !important;
    }

    .stTitle, .stSubheader, .stHeader, .stText, .stTextInput label, .stFileUploader label, .stTextArea label {
        color: black !important;
    }

    .stWarning, .stError, .stInfo {
        background-color: #cce5ff !important;
        color: black !important;
    }

    .stSidebar, .stSidebar .sidebar-content {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Título de la app
st.title('RAG - Chatea con un PDF!')
st.write("Versión de Python:", platform.python_version())
st.write("Mediante esta app podrás chatear con un bot que se leyó un PDF que le proporciones.")

# Cargar imagen
try:
    image = Image.open('robotcito.png')
    st.image(image, width=500)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Input para clave de API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Cargar PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesar el PDF
if pdf is not None and ke:
    try:
        # Leer el PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Dividir el texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Pregunta del usuario
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
