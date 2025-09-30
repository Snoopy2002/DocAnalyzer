# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:44:59 2025

@author: Snoopy
"""
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


def load_document(file): #loads file types
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings=OpenAIEmbeddings()
    vector_store=Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k): #k most similar chunks of data to answer a question you ask, higher k is more elaborate, and costs more
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4o', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer

def calc_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    #print(f'Total Tokens: {total_tokens}')
    #print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return(total_tokens, total_tokens/1000*.0004)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    #f=open('chatkey.txt','r')
    #key=f.readline()
    #print(key)
    #os.environ["OPENAI_API_KEY"]=key
    #os.getenv('OPENAI_API_KEY')
    
    st.image('parrot.png')
    st.subheader('LLM ? Answering Application')
    with st.sidebar:
        api_key=st.text_input('Enter your OpenAI Key: ', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY']=api_key
        uploaded_file=st.file_uploader('Upload file: ', type=['pdf','docx','txt'])
        chunk_size=st.number_input('Chunk size (smaller=more details):',min_value=100, max_value=2048, value=512, on_change=clear_history)
        k=st.number_input('k (similar chunks, greater = more detail)', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Analyze file', on_click=clear_history)
        
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file: '):
                bytes_data = uploaded_file.read()
                file_name=os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                    
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                tokens, embedding_cost = calc_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                vector_store = create_embeddings(chunks) #save vector store btn page loads
                st.session_state.vs = vector_store
                st.success ('File upload, embed and chunk success!')
                
    question=st.text_input('Ask a question about the content of your file: ')
    if question:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            #st.write(f'k:{k}')
            answer = ask_and_get_answer(vector_store, question, k)
            st.text_area('LLM Answer: ', value=answer)
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {question} \n A: {answer}'
            st.session_state.history = f'{value}\n{"-"*100} \n {st.session_state.history}'
            h=st.session_state.history

            st.text_area(label='Chat History', value=h, key='history', height=600)




