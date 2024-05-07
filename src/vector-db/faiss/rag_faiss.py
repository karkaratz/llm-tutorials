from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

load_dotenv()
# Full path and file_name for your FAISS index.
index_name = "index_faiss"

if __name__ == "__main__":
    print("Job Started")

    embeddings = OpenAIEmbeddings( # Embeddings configuration. In my case I am using a local LM Studio instance with nomic ai text embedding.
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        model="gpt2",
        tiktoken_enabled=False
    )

    llm = OpenAI( # LLM instance creation. I am using a local LM Studio model that is exporting OpenAI compatible APIs.
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0.1
    )

    print(llm.generate(["Who is Poorpoorpoorman?"])) # Print the answer to a query without knowledge.
    new_vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
        llm=llm,  chain_type="stuff", retriever=new_vectorstore.as_retriever(search_kwargs={'k': 10 })
    )
    res=qa.invoke("If you do not know, do not answer. Use only the context I am providing. Who is Poorpoorpoorman?") # Print the anser with RAG
    print(res)