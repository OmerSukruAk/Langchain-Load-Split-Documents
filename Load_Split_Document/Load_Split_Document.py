from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
import logging 
import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the directory of this file

if os.path.exists('Load&Split.LOG'):    # If the log file exists, delete it
    os.remove('Load&Split.LOG')
logging.basicConfig(format = '%(levelname)s:%(message)s', filename='Load&Split.LOG', level=logging.INFO)

logging.info("Loading PDF file...")
loader = PyPDFLoader("..\Human Activity Recognition.pdf")

pages = loader.load() # Load the PDF file
logging.critical("PDF loaded successfully.")
logging.info(str(pages[0]).replace("\n", " "))

logging.info("Splitting text into chunks (Character Text Splitter)...")
text_splitter = CharacterTextSplitter(chunk_size = 150, chunk_overlap = 20, separator="\n") # Create a CharacterTextSplitter object

logging.info("Splitting text into documents...")
docs = text_splitter.split_documents(pages) # Split the text into documents
logging.critical("Text split into " + str(len(docs)) + " documents.")


logging.info(str(docs[0]).replace("\n", " "))


logging.info("Splitting text into chunks... (Token Text Splitter)")
text_splitter = TokenTextSplitter(chunk_size=30, chunk_overlap=10) # Create a TokenTextSplitter object
tokens = text_splitter.split_documents(docs) # Split the text into tokens
logging.critical("Text split into " + str(len(tokens)) + " chunks.")

logging.info(str(tokens[0]).replace("\n", " "))