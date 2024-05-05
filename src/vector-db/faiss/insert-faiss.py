from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()
# Path of the document to read (This example explores the loading of a single document)
pdf_path = "<YOUR_FILE_PATH_HERE>"

# Full path and file_name for your FAISS index.
save_path = "<YOUR_FILE_PATH_HERE/YOUR_FILENAME"

# How to split and embed the text
chunk_size = 300    # change with your size
chunk_overlap = 10  # change with your overlap size.

# Connection to an embedding engine that is supporting OpenAI apis.
# In this example, I'm using LM Studio with nomic-ai/nomic-embed-text-v1.5-GGUF
# If you want to run it against standard OpenAI embeddings, use the standard OpenAI configuration.
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",    # My LM studio is running on localhost, port 1234.
    api_key="not-needed",   # My LM Studio does not require a password.
    model="gpt2",           # Just a string. My LM Studio does not require to specify the model. Needed in other cases.
    tiktoken_enabled=False
)

if __name__ == "__main__":

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

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(save_path)