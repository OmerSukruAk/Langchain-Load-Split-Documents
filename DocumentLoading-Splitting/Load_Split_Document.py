from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import logging 
import os

if os.path.exists('Load&Split.LOG'):
    os.remove('Load&Split.LOG')
logging.basicConfig(format = '%(levelname)s:%(message)s', filename='Load&Split.LOG', level=logging.INFO)

logging.info("Loading PDF file...")
loader = PyPDFLoader("Human Activity Recognition.pdf")

pages = loader.load()
logging.critical("PDF loaded successfully.")
logging.info(str(pages[0]).replace("\n", " "))

logging.info("Splitting text into chunks (Character Text Splitter)...")
text_splitter = CharacterTextSplitter(chunk_size = 150, chunk_overlap = 20, separator="\n")

logging.info("Splitting text into documents...")
docs = text_splitter.split_documents(pages)
logging.critical("Text split into " + str(len(docs)) + " documents.")


logging.info(str(docs[0]).replace("\n", " "))

from langchain.text_splitter import TokenTextSplitter

logging.info("Splitting text into chunks... (Token Text Splitter)")
text_splitter = TokenTextSplitter(chunk_size=30, chunk_overlap=10)
tokens = text_splitter.split_documents(docs)
logging.critical("Text split into " + str(len(tokens)) + " chunks.")

logging.info(str(tokens[0]).replace("\n", " "))