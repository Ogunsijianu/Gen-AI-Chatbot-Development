import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from transformers import BertTokenizer
from htmlTemplates import css, bot_template, user_template



# Load PDF
loaders = [
    PyPDFLoader("CGC-Aquatics-Programs-Parent-Handbook.pdf"),
    PyPDFLoader("client-handbook-recreation.pdf"),
    PyPDFLoader("curriculum_swimming_and_water_safety-a_guide_for_parents.pdf"),
    PyPDFLoader("dipadees_learntoswim_infobk.pdf"),
    PyPDFLoader("LTS flyer Term 3 2015.pdf"),
    PyPDFLoader("National-Swimming-and-Water-Safety-Framework_FINAL-2020.pdf"),
    PyPDFLoader("Parent Handbook.pdf"),
    PyPDFLoader("Scientific Advisory Council SCIENTIFIC REVIEW - Minimum Age for Swimming Lessons.pdf"),
    PyPDFLoader("ssa_info_book.pdf"),
    PyPDFLoader("Swim+School+-+Parents+Handbook-lowres-web.pdf"),
    PyPDFLoader("Swimming Lessons Level Progression Chart _ The Y.pdf"),
    PyPDFLoader("Swim-Lessons-Parent-Handbook_r4.pdf"),
    PyPDFLoader("WIRAC LTS Timetable.pdf"),
    PyPDFLoader("Y NSW Swim School Program - Terms and Conditions _ The Y.pdf"),
    PyPDFLoader("YMCA Swim Lesson Level Guide.pdf"),
    PyPDFLoader("YMCA-Swim-School-Brochure.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_tokenizer(text_chunks):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_token = tokenizer.tokenize(str(text_chunks))
    return text_token



def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()