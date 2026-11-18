# load_data.py (Claude version): load, clean, tag, chunk, embed and store the book
import json
import os
import time
from typing import List

import anthropic
from pydantic import BaseModel
import certifi
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import key_param

# Where the chunks will live
# certifi supplies trusted certificates, which avoids CERTIFICATE_VERIFY_FAILED on some Macs
client = MongoClient(key_param.MONGODB_URI, tlsCAFile=certifi.where())
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]
client.admin.command("ping")   # fail fast if Atlas can't be reached
print("Connected to MongoDB Atlas")

# 1. Load
loader = PyPDFLoader("./sample_files/mongodb.pdf")
pages = loader.load()

# 2. Sanitize: keep pages with real text
cleaned_pages = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)
print(f"Loaded {len(pages)} pages, kept {len(cleaned_pages)}")


# 3. Tag each page with metadata using Claude structured outputs
class PageMetadata(BaseModel):
    title: str
    keywords: List[str]
    hasCode: bool


claude = anthropic.Anthropic(api_key=key_param.anthropic_api_key)


def tag_page(page):
    response = claude.beta.messages.parse(
        model="claude-opus-5-5",
        max_tokens=2000,
        output_config={"effort": "low"},          # a simple extraction job
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                      # retry on another model if a request is declined
        messages=[{
            "role": "user",
            "content": (
                "Read this page from a book about MongoDB and extract: a short title, "
                "a list of keywords, and whether the page contains code (hasCode).\n\n"
                + page.page_content
            ),
        }],
        output_format=PageMetadata,
    )
    page.metadata.update(response.parsed_output.model_dump())
    return page


# Tagging costs money and time, so the results are saved to tagged_pages.json.
# Re-runs reuse that file instead of asking Claude again.
# (Delete tagged_pages.json if you change the PDF and want fresh tags.)
TAGS_FILE = "tagged_pages.json"
if os.path.exists(TAGS_FILE):
    with open(TAGS_FILE) as f:
        docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in json.load(f)]
    print(f"Using saved tags for {len(docs)} pages from {TAGS_FILE}")
else:
    docs = []
    for i, p in enumerate(cleaned_pages, start=1):
        print(f"Tagging page {i} of {len(cleaned_pages)}...")
        docs.append(tag_page(p))
    with open(TAGS_FILE, "w") as f:
        json.dump([{"text": d.page_content, "metadata": d.metadata} for d in docs], f)
    print(f"Saved tags to {TAGS_FILE}")

# 4. Chunk the tagged pages (each chunk inherits its page's metadata)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 5. Embed with Voyage AI and store in Atlas
embeddings = VoyageAIEmbeddings(
    model="voyage-4",                         # 1,024 dimensions
    voyage_api_key=key_param.voyage_api_key,
)

print("Creating embeddings and saving to Atlas...")
collection.delete_many({})   # start fresh so re-runs don't create duplicates

vectorStore = MongoDBAtlasVectorSearch(
    collection=collection, embedding=embeddings, index_name="vector_index"
)

# Send the chunks to Voyage AI in small batches. If Voyage says "slow down"
# (a rate limit), wait and try the same batch again instead of crashing.
BATCH = 20
for start in range(0, len(split_docs), BATCH):
    batch = split_docs[start:start + BATCH]
    while True:
        try:
            vectorStore.add_documents(batch)
            break
        except Exception as e:
            if "RateLimit" not in type(e).__name__:
                raise
            print("Voyage AI rate limit reached. Waiting 60 seconds, then retrying...")
            time.sleep(60)
    print(f"Saved chunks {start + 1} to {start + len(batch)} of {len(split_docs)}")

print(f"Stored {collection.count_documents({})} chunks in {dbName}.{collectionName}")
