import hashlib

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import chromadb


load_dotenv()
# Path of the document to read (This example explores the loading of a single document)
pdf_path = "/Users/vincenzo/dev_projects/faiss-test/dama-dmbok-test.pdf"
# The collection to use in chromadb
collection_name = "langflow-test2"
# Information for chromadb http requests
chroma_host = "localhost"
chroma_port = 8000
# How to split and embed the text
chunk_size = 300
chunk_overlap = 10

# Connection to an embedding engine that is supporting OpenAI apis.
# In this example, I'm using LM Studio with nomic-ai/nomic-embed-text-v1.5-GGUF
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1", # My LM studio is running on localhost, port 1234
    api_key="not-needed", # My LM Studio does not require a password.
    model="gpt2", # Just a string. My LM Studio does not require to specify the model. Needed in other cases.
    tiktoken_enabled=False
)

if __name__ == "__main__":
    # Create a Chroma DB Http Client
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    # list all collections
    print(client.list_collections())
    # Using the client, get an instance of a specific collection.
    # If the collection does not exists, it creates a new collection.
    collection = client.get_or_create_collection(collection_name)
    print("Job Started")

    # Instantiation a Loader specialized on PDF files.
    loader = PyPDFLoader(pdf_path)
    # The variable documents will hold the content of your PDF file as a single item.
    documents = loader.load()

    # The splitter will transform the content of the pdf, splitting it on a several chunks.
    #
    # chunk_size refers to the number of tokens or characters each text segment (chunk) should
    #   contain when splitting a larger text into smaller, manageable pieces.
    #   This size is determined based on the maximum input size limit of the LLM.
    #   For instance, if a model can process a maximum of 512 tokens at once, setting the chunk_size to
    #   or below 512 ensures that each segment of the text fits within this limit.
    #
    # chunk_overlap refers to the number of tokens or characters that consecutive chunks will share.
    #   This parameter is crucial for ensuring that there isn't any loss of context between chunks,
    #   which is particularly important for tasks that require understanding of context, like text generation
    #   or complex sentiment analysis.

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    # docs will contain a List[Documents] where every document will be a dictionary containing:
    #   page_content : The text of a chunk.
    #   metadata : Metadata of the original pfd, like document path and page.
    docs = text_splitter.split_documents(documents=documents)
    print(len(docs)) # a simple print of how many chunks we have generate

    # Preparing for writing into vector db. Chroma in our example.
    # Just remember: an item written into a vector database is calle document.
    # Creating 4 lists
    #   1 page_contents - The original content of chunks
    #   2 ids - unique identifier of our documents
    #   3 metadata - metadata gathered from the original pdf file
    #   4 x - embedding results
    page_contents = [doc.page_content for doc in docs if doc.page_content]
    ids = [hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest() for doc in docs if doc.page_content]
    metadata = [doc.metadata for doc in docs if doc.page_content]
    x = embeddings.embed_documents(page_contents, 300)
    # write everything into the vector db, using the collection object gathered from chroma client.
    collection.upsert(
        documents=page_contents,
        embeddings=x,
        metadatas=metadata,
        ids=ids,
    )
    print("Completed")
