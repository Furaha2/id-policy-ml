from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import streamlit as st

# streamlit app config
st.set_page_config(page_title="ID Policy Chatbot")
st.title("ID Policy Bot")

# input openai key
api_key = st.sidebar.text_input('OpenAI API Key')

# save env vars as secrets
# access env variables
db_uri = st.secrets["mongo_db_uri"]
db_name = st.secrets["db_name"]
collection_name = st.secrets["collection_name"]
atlas_vector_search_index = st.secrets["search_index"]
openai_key = st.secrets["openai_key"]

model = 'gpt-4o'
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        db_uri,
        db_name + "." + collection_name,
        OpenAIEmbeddings(),
        index_name=atlas_vector_search_index)


# load embeddings from mongo-db
retriever = vector_search.as_retriever(search_kwargs={'k':10})


# generate queries
prompt_q = ChatPromptTemplate.from_template(
    """
    You are an intelligent assistant. Your task is to generate at least 6 questions based on the provided question in different wording and different perspectives to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
    In case the question is asking similarities from more than one country then don't generate 6 questions, just generate same question for each of the countries mentioned. Original question: {question}  
"""
)

# generate questions
generate_questions = (
    {"question": RunnablePassthrough()}
    | prompt_q
    | ChatOpenAI(model=model, temperature=0.7)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# testing how generate queries work
question = ""

def get_context_union(docs: List[List]):
    all_docs = [d.page_content for doc in docs for d in doc] # get page content only
    distinct_docs = list(set(all_docs))
    # extract page content only
    return distinct_docs

rag_chain = (
    {'question': RunnablePassthrough()}
    | generate_questions
    | retriever.map() # applies retriever to each generated question
    | get_context_union # aggregates the output from previous step together
)

# implement final step of multi-query to get answers
prompt_final = ChatPromptTemplate.from_template(
    """
    Asnwer the given question using the provided context.\n\nContext: {context}\n\nQuestion: {question} 
    """
)

llm = ChatOpenAI(model=model, temperature=0)

multi_query_chain = (
    {'context': rag_chain, 'question': RunnablePassthrough()} # context is answer returned above
    | prompt_final
    | llm
    | StrOutputParser()
)

multi_query_chain.invoke(question)