# rag.py (Claude version): answer questions about MongoDB using The Little MongoDB Book
import anthropic
import certifi
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import key_param

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

# certifi supplies trusted certificates, which avoids CERTIFICATE_VERIFY_FAILED on some Macs
client = MongoClient(key_param.MONGODB_URI, tlsCAFile=certifi.where())

vectorStore = MongoDBAtlasVectorSearch(
    collection=client[dbName][collectionName],
    embedding=VoyageAIEmbeddings(
        model="voyage-4",                     # must match load_data.py
        voyage_api_key=key_param.voyage_api_key,
    ),
    index_name=index,
)

claude = anthropic.Anthropic(api_key=key_param.anthropic_api_key)


def ask_claude(prompt_value):
    """Send the filled-in prompt to Claude and return only the answer text."""
    response = claude.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt_value.to_string()}],
    )
    if response.stop_reason == "refusal":
        return "I can't answer that."
    return "".join(block.text for block in response.content if block.type == "text")


def query_data(query):
    # Retrieve: find the 3 chunks closest in meaning to the question
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Prompt: ground the model in the retrieved context only
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    Context:
    {context}
    Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }

    # Generate: Claude takes the place of the OpenAI chat model
    llm = RunnableLambda(ask_claude)
    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )

    answer = rag_chain.invoke(query)
    return answer


if __name__ == "__main__":
    print(query_data("What is the difference between a collection and database in MongoDB?"))
    print(query_data("Why is the sky blue?"))
