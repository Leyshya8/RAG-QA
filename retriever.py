from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Custom prompt — instructs the LLM to use ONLY the retrieved context
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

def load_qa_chain(index_path: str = "./faiss_index"):

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str, chain=None, retriever=None):
    if chain is None:
        chain, retriever = load_qa_chain()

    answer = chain.invoke(question)

    # Get sources separately
    source_docs = retriever.invoke(question)
    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200]
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}
# Quick test
if __name__ == "__main__":
    response = ask("What is this document about?")
    print("Answer:", response["answer"])
    print("\nSources:")
    for s in response["sources"]:
        print(f"  - {s['source']} (page {s['page']}): {s['snippet']}...")