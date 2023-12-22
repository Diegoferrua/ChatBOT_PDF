from PyPDF2 import PdfReader
import os

## LIBRERIAS LANGCHAIN
from langchain.text_splitter import CharacterTextSplitter # Splittear el text
from langchain.embeddings.openai import OpenAIEmbeddings #embebidos
from langchain.vectorstores import FAISS #vectorizacion

##Librerias para el Textconteiner
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


from streamlit_chat import message

## configurando la plantilla
from langchain import PromptTemplate

import streamlit as  st

## Configurando streamlit
st.set_page_config(page_title="ChatBot con PDf", layout="wide")
st.markdown("""<style>,block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)


## Set OpenAI API KEY
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hola! En que puedo ayudarte?"]

if 'requests' not in st.session_state:
    st.session_state['requests']=[]


#funcion para las bases caracteristicas
def create_embeddings(pdf):
    #Extrayendo text del pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text =""

        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap=200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        embeddings_pdf = FAISS.from_texts(chunks, embeddings)

        return embeddings_pdf
#interfaz



#Cargar el documento PDf en el SIDEBAR
st.sidebar.markdown("<h1 style='text-align:center; color: #176887;'>Cargar Archivo PDF</h1>", unsafe_allow_html=True)
st.sidebar.write("Carga el archivo PDF con el que quieres interactuar")
pdf_doc = st.sidebar.file_uploader("", type="pdf")
st.sidebar.write("---")
#clear_button = st.sidebar.button("Limpiar Conversacion", key="clear")


embeddings_pdf = create_embeddings(pdf_doc)

#Chat Session

st.markdown("<h2 style= 'text-align: center; color: #176B87; text:decoration: underline;'><strong>Interactua con el bot</strong></h2>", unsafe_allow_html=True)
st.write("---")

response_container = st.container()

textcontainer = st.container()

## promtp template


##template de la respuesta
prompt_template = """Responda la pregunta con la mayor precisión posible utilizando el contexto proporcionado. si la respuesta no está 
                    contenida en el contexto, digamos "La pregunta está fuera de contexto, 'no me enseñaron ello ☹️' " \n\n
                    contexto: \n {context}?\n
                    pregunta: \n {question} \n
                    respuesta:
                  """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


with textcontainer:

    with st.form(key='my_form',clear_on_submit=True):
        query = st.text_area("Tu: ", key='input', height=100 )
        submit_button = st.form_submit_button(label='Enviar')


    if query:
        with st.spinner("Escribiendo..."):
            docs = embeddings_pdf.similarity_search(query)

            llm = OpenAI(model_name = "text-davinci-003")

            chain = load_qa_chain(llm,chain_type ="stuff",prompt=prompt)

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

## configurando el campo response_container
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i),avatar_style="pixel-art")
            if i<len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True,key=str(i)+'_user')