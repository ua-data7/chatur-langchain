from typing import Optional, List, Literal
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

import tiktoken


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def create(docs:List[Document], persist_db_path:Optional[str]=None) -> Chroma:
    # store it into vector store (chroma) using gpt4all embeddings
    return Chroma.from_documents(
        documents=docs,
        embedding=GPT4AllEmbeddings(),
        persist_directory=persist_db_path)


# this splits the input text
text_splitter = RecursiveCharacterTextSplitter(
    # chunk size should not be very large as model has a limit
    chunk_size = 1000,
    # this is a configurable value
    chunk_overlap = 200,
    length_function = tiktoken_len,
)

def create_from_text(text:Literal) -> Chroma:
    # chunk input into multiple documents
    docs = text_splitter.create_documents([text])
    return create(docs)

def create_from_pdf(pdf_path:str) -> Chroma:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    return create(docs)

def create_from_markdown(markdown_path:str) -> Chroma:
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    return create(docs)

def create_from_pptx(pptx_path:str) -> Chroma:
    loader = UnstructuredPowerPointLoader(pptx_path)
    docs = loader.load()
    return create(docs)
