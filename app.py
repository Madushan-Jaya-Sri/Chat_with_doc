import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import base64

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    context:\n{context}\n
    question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Ask DocMan 🤖 ")

   
   # Encode the image to base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_path = os.path.join("images", "Firefly give me an image of Artificial Intelligent bot searching for something document files, use d (2).jpg")
    base64_img = get_base64_image(img_path)

    # Use base64 image as background
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    headerbg = f"""
    <style>
    [data-testid="stHeader"] {{
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(headerbg, unsafe_allow_html=True)

    # Inject custom CSS to change the placeholder color to #339988
    st.markdown(
        """
        <style>
        /* Change the placeholder color */
        .stTextInput input::placeholder {
            color: #339988;
        }
        /* Change the text color */
        .stTextInput input {
            color: orange;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
