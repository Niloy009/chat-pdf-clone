import os
import pickle
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


with st.sidebar:
    st.title(body='This is Chat-Pdf clone app using LLM')
    st.markdown(body='''
                ## About
                This app is a clone version of [Chat-Pdf](https://www.chatpdf.com/) using:
                - [Streamlit](https://www.streamlit.io/)
                - [LangChain](https://python.langchain.com/docs/get_started/introduction)
                - [OpenAI](https://platform.openai.com/docs/models) LLM mdoel
                ''')
    add_vertical_space(num_lines=5)
    st.write('Made with 💘 by [Niloy Saha Roy](https://www.github.com/Niloy009)')

def main():
    st.header(body="Chat with PDF ☁️")
   
    # Upload a pdf file
    pdf = st.file_uploader(label='## Upload your Pdf', type='pdf')
    model_id = "mdeberta-v3-base-squad2"
    
    if pdf is not None:
        pdf_reader = PdfReader(stream=pdf)
        
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
    
        if os.path.exists(path=f'{store_name}.pkl'):
            with open(file=f'{store_name}.pkl', mode="rb") as f:
                VectorStore = pickle.load(file=f)
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model_id,
                                       model_kwargs={'device': 'cuda:0'})
            VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            with open(file=f'{store_name}.pkl', mode='wb') as f:
                pickle.dump(obj=VectorStore,file=f)
        query = st.text_input('Enter Your Question:')
        if query is not None:
            ollama = Ollama(base_url='http://localhost:11434',
                            model="llama2")
            docs = VectorStore.similarity_search(query, k=3)
            qachain=RetrievalQA.from_chain_type(ollama, retriever=VectorStore.as_retriever())
            response = qachain({"query": query})
            st.write(f"Question: {response['query']}")
            st.write(f"Answer: {response['result']}")

    else:
        st.write('## Please first upload a pdf')

    
    
        
if __name__ == '__main__':
    main()
    
    
# Embedding model from huggingfaces LLAMA2 provided by meta