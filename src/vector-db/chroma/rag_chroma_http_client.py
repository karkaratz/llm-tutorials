from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chromadb

load_dotenv()

# This example uses a chroma db that has been loaded with Poorpoorpoorman.pdf"
# If you have not done it yet, use insert-pdf-chromadb-http-client.py to upload the pdf to your chroma db.

# The collection to use in chromadb
collection_name = "<YOUR_COLLECTION_HERE>"
# Information for chromadb http requests
chroma_host = "<YOUR_PORT_HERE>" # In my example it is localhost
chroma_port = 8000  # Change with your port
# How to split and embed the text
chunk_size = 300    # change with your size
chunk_overlap = 10  # change with your overlap size.


embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",    # My LM studio is running on localhost, port 1234
    api_key="not-needed",   # My LM Studio does not require a password.
    model="gpt2",           # Just a string. My LM Studio does not require to specify the model. Needed in other cases.
    tiktoken_enabled=False
)

llm = OpenAI(
    base_url="http://localhost:1234/v1",    # My LM studio is running on localhost, port 1234
    api_key="not-needed",   # My LM Studio does not require a password.
    model="gpt2"           # Just a string. My LM Studio does not require to specify the model. Needed in other cases.
)


if __name__ == "__main__":
    # Create a Chroma DB Http Client
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # list all collections
    print(client.list_collections())
    # Using the client, get an instance of a specific collection.
    # If the collection does not exists, it creates a new collection.
    collection = client.get_or_create_collection(collection_name)
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm) # This is the updated way to call an llm, using Langchain.

    qs = [
         {'question': "What is Kaggle?"},
         {'question': "What is the first step I should take on Kaggle?"},
         {'question': "I followed your instructions. What should I do next?"},
         {'question': "Who is Poorpoorpoorman?"} # Asking the question without RAG
     ]

    for i in qs:
        print(llm_chain.invoke(i)) # Invoking the llm with langchain for every questions in qs.

    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains import RetrievalQA

    db = Chroma(client=client, collection_name=collection_name,embedding_function=embeddings) # generating a db and Retriever instance from Chroma DB
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 1})
    prompt = hub.pull("rlm/rag-prompt") # Loading a predefined rag-prompt

    rag_chain = ( # the rag chain uses a retriever to provide a context then concatenates it with the prompt and send the result to llm
            {"context": retriever , "question": RunnablePassthrough()}
            | prompt
            | llm
    )
    print(rag_chain.invoke("Who is Poorpoorpoorman.")) # Asking the question with RAG, the model, now, knows about Poorpoorpoorman.