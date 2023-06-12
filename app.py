# from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def main():
    # load the .env file
#     load_dotenv()

    # set up the user interface.
    st.set_page_config(page_title="Ask your pdf")
    st.title("Ask your pdf ✔")

    # the following will get the pdf file.
    pdf = st.file_uploader(label="Upload your pdf", type="pdf")

    # extracting the text from the pdf file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # divide the whole text into smaller chunks
        text_splitter = CharacterTextSplitter(separator="\n", length_function=len, chunk_size=1000, chunk_overlap=200)

        chunks = text_splitter.split_text(text)

        # create embeddings for chunks
        embeddings = OpenAIEmbeddings()

        # create the knowledgebase
        knowledgebase = FAISS.from_texts(chunks, embeddings)

        # now we will allow the user to ask questions.
        user_input = st.text_input("Ask any question from your pdf: ")
        if user_input:
            docs = knowledgebase.similarity_search(user_input)
            llm = OpenAI()

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_input)
            st.write(response)
            


        
        



if __name__ == "__main__":
    main()
