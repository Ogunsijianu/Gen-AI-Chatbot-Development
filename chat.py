# Imports
import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


os.environ.get('OPENAI_API_KEY')


st.set_page_config(page_title="Chatbot", page_icon=":books:")
st.title("GreeneDesk Chatbot Application")
# Load PDF documents for training
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

text_chunks = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# create embeddings for each text chunk using the FAISS class, which creates a vector index using FAISS and allows efficient searches between vectors
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings) 


  # Function to create conversation chain
def get_conversation_chain(vector_store):
    # Import the neural language model using the LlamaCpp class, which allows you to use a GPT-3 model in C++ with various parameters such as temperature, top_p, verbose and n_ctx (maximum number of tokens that can be generated)

    llm = ChatOpenAI()

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function
def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    
    with st.spinner('Processing'):
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vector_store)

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)
    
    
if __name__ == '__main__':
    main()
