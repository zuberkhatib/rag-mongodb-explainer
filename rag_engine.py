"""
rag_engine.py - the "engine room" of the demo.

Everything in here runs WITHOUT any API keys, so the app works for anyone on
Streamlit Community Cloud. It imitates each stage of the real RAG app:

    real app (load_data.py / rag.py)          this demo
    --------------------------------          -----------------------------
    PyPDFLoader on mongodb.pdf                DEMO_PAGES (short sample pages)
    word-count sanitizer (> 20 words)         sanitize()  (same rule)
    Claude metadata tagging                   pre-filled tags on each page
    RecursiveCharacterTextSplitter(500,150)   split_text() (same idea)
    Voyage AI embeddings (1,024 numbers)      TF-IDF keyword vectors
    Atlas Vector Search ($vectorSearch)       cosine similarity in numpy
    Claude writes the answer                  extractive answer (or Claude
                                              when keys are set in secrets)

The optional "live" helpers at the bottom call the real services when the app
owner adds keys to Streamlit secrets.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 1. Demo "book": short pages written for this demo, in the spirit of
#    The Little MongoDB Book by Karl Seguin (CC BY-NC 3.0), which the real app
#    loads. Tags are what an LLM tagger would produce (title, keywords, hasCode).
# ---------------------------------------------------------------------------

DEMO_PAGES = [
    {
        "page": 0,
        "text": "",
        "title": "Cover",
        "keywords": [],
        "hasCode": False,
    },
    {
        "page": 1,
        "text": "The Little MongoDB Book. Licensed for free use with attribution.",
        "title": "Title page",
        "keywords": ["license"],
        "hasCode": False,
    },
    {
        "page": 5,
        "text": (
            "The Basics. MongoDB is built from a handful of simple ideas. A MongoDB "
            "server can hold zero or more databases, and each database acts as a "
            "high-level container for everything else.\n\n"
            "A database holds zero or more collections. A collection shares enough "
            "with a traditional table that you can think of the two as the same thing. "
            "Collections are made up of zero or more documents, and a document can be "
            "thought of as a row. A document is made of one or more fields, which are "
            "a lot like columns.\n\n"
            "Indexes in MongoDB work much like indexes in relational databases. Cursors "
            "are what you get back when you ask MongoDB for data: a pointer to the "
            "results that lets you count or skip ahead before any data is read."
        ),
        "title": "The Basics",
        "keywords": ["database", "collection", "document", "field", "index", "cursor"],
        "hasCode": False,
    },
    {
        "page": 9,
        "text": (
            "Mastering Find. The find command takes a selector, a JSON object that "
            "describes which documents you want. The empty selector {} matches every "
            "document in a collection.\n\n"
            "db.unicorns.find({gender: 'm', weight: {$gt: 700}})\n\n"
            "Operators such as $lt, $lte, $gt, $gte and $ne compare values, $exists "
            "checks whether a field is present, $in matches one of several values and "
            "$or combines conditions. A second argument, the projection, picks which "
            "fields come back: db.unicorns.find({}, {name: 1, _id: 0})."
        ),
        "title": "Mastering Find",
        "keywords": ["find", "selector", "query operators", "projection", "$gt", "$or"],
        "hasCode": True,
    },
    {
        "page": 14,
        "text": (
            "Updating. In its simplest form update takes a selector and a change. Use "
            "the $set operator to change some fields without replacing the whole "
            "document:\n\n"
            "db.unicorns.updateOne({name: 'Pilot'}, {$set: {weight: 590}})\n\n"
            "Other modifiers include $inc to add to a number and $push to append to an "
            "array. An upsert updates the document if it is found and inserts it if it "
            "is not, which is handy for counters: pass {upsert: true} as the third "
            "argument."
        ),
        "title": "Updating",
        "keywords": ["update", "$set", "$inc", "$push", "upsert"],
        "hasCode": True,
    },
    {
        "page": 21,
        "text": (
            "Data Modeling. MongoDB has no joins in the relational sense, so modeling "
            "starts by asking how the data will be read. Data that is read together is "
            "often stored together by embedding one document inside another, such as "
            "an address inside a customer.\n\n"
            "When related data grows without limit, store a reference instead and look "
            "it up. The $lookup stage of the aggregation pipeline can combine documents "
            "from two collections, which covers many join-like needs. A flexible schema "
            "means documents in the same collection do not need identical fields."
        ),
        "title": "Data Modeling",
        "keywords": ["data modeling", "embedding", "references", "$lookup", "schema"],
        "hasCode": False,
    },
    {
        "page": 34,
        "text": (
            "Transactions. For a long time MongoDB offered atomic writes on a single "
            "document only, and developers handled multi-step changes with patterns "
            "such as two-phase commit.\n\n"
            "MongoDB 4.0, released in 2018, added multi-document ACID transactions on "
            "replica sets, and version 4.2 extended them to sharded clusters. You start "
            "a transaction, make your changes and then commit or abort, just as you "
            "would in a relational database. Drivers retry automatically on retryable "
            "errors. The addition of transactions addressed a serious concern, so "
            "MongoDB now sits right in the middle of the data storage landscape."
        ),
        "title": "Transactions",
        "keywords": ["transactions", "ACID", "multi-document", "atomicity", "MongoDB 4.0"],
        "hasCode": False,
    },
    {
        "page": 40,
        "text": (
            "Performance and Indexes. An index lets MongoDB find documents without "
            "scanning a whole collection. Create one with createIndex and check how a "
            "query runs with explain:\n\n"
            "db.unicorns.createIndex({name: 1})\n"
            "db.unicorns.find({name: 'Pilot'}).explain()\n\n"
            "If explain reports COLLSCAN, every document was read; IXSCAN means an index "
            "was used. Compound indexes cover several fields, and the order of fields "
            "matters for sorting."
        ),
        "title": "Performance and Indexes",
        "keywords": ["index", "createIndex", "explain", "COLLSCAN", "IXSCAN", "performance"],
        "hasCode": True,
    },
    {
        "page": 46,
        "text": (
            "The Aggregation Pipeline. Aggregation transforms and summarizes documents "
            "by passing them through stages, a bit like an assembly line. $match "
            "filters documents, $group totals them, $sort orders them and $project "
            "reshapes them.\n\n"
            "Each stage hands its output to the next one. Atlas Vector Search is also "
            "used as a stage: $vectorSearch must be the first stage, and it returns the "
            "documents whose embeddings are closest to a query vector."
        ),
        "title": "The Aggregation Pipeline",
        "keywords": ["aggregation", "pipeline", "$match", "$group", "$vectorSearch"],
        "hasCode": False,
    },
    {
        "page": 52,
        "text": (
            "Replication and Sharding. A replica set is a group of servers that keep "
            "copies of the same data. One primary accepts writes and secondaries copy "
            "them, so if the primary fails an election picks a new one and the "
            "application keeps running.\n\n"
            "Sharding splits a large collection across many servers using a shard key, "
            "so a database can grow beyond the limits of a single machine. Replication "
            "is about availability; sharding is about scale."
        ),
        "title": "Replication and Sharding",
        "keywords": ["replica set", "primary", "secondary", "sharding", "shard key"],
        "hasCode": False,
    },
    {
        "page": 58,
        "text": (
            "When to Use MongoDB. MongoDB fits well when your data does not have a fixed "
            "shape, when you want to change your schema without migrations, or when you "
            "need to scale writes across many servers.\n\n"
            "It also offers geospatial indexes for location queries, full-text search "
            "and vector search in Atlas for AI applications such as retrieval augmented "
            "generation. As with any tool, pick it because it suits your data and access "
            "patterns rather than out of habit."
        ),
        "title": "When to Use MongoDB",
        "keywords": ["use cases", "flexible schema", "geospatial", "vector search", "scaling"],
        "hasCode": False,
    },
]

SOURCE = "./sample_files/mongodb.pdf"


# ---------------------------------------------------------------------------
# 2. Sanitize: the exact rule from load_data.py
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    # Same test as the real app: len(page.page_content.split(" "))
    return len(text.split(" ")) if text else 0


def sanitize(pages: list[dict], min_words: int = 20) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for p in pages:
        (kept if word_count(p["text"]) > min_words else dropped).append(p)
    return kept, dropped


# ---------------------------------------------------------------------------
# 3. Chunk: a small version of LangChain's RecursiveCharacterTextSplitter.
#    Try to split on paragraphs, then lines, then spaces, then characters,
#    and glue pieces back together up to chunk_size with chunk_overlap.
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", " ", ""]


def _atoms(text: str, chunk_size: int, separators: list[str]) -> list[tuple[str, str]]:
    """Cut text into pieces no longer than chunk_size.
    Each piece remembers the separator that joins it to the piece before it."""
    if len(text) <= chunk_size:
        return [(text, "")]
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            rest = separators[i + 1:]
            break
    parts = list(text) if sep == "" else text.split(sep)
    out: list[tuple[str, str]] = []
    for part in parts:
        if not part:
            continue
        sub = [(part, "")] if len(part) <= chunk_size or not rest else _atoms(part, chunk_size, rest)
        first = True
        for piece, inner_sep in sub:
            out.append((piece, sep if first else inner_sep))
            first = False
    return out


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 150) -> list[str]:
    """Glue pieces back together up to chunk_size, repeating up to
    chunk_overlap characters from the end of one chunk at the start of the next."""
    if not text:
        return []
    chunk_overlap = min(chunk_overlap, chunk_size - 1)

    def join(parts):
        return "".join((s if i else "") + p for i, (p, s) in enumerate(parts))

    chunks: list[str] = []
    window: list[tuple[str, str]] = []
    for piece, sep in _atoms(text, chunk_size, SEPARATORS):
        if window and len(join(window + [(piece, sep)])) > chunk_size:
            chunks.append(join(window).strip())
            # drop pieces from the front until what is left fits as overlap
            while window and (len(join(window)) > chunk_overlap
                              or len(join(window + [(piece, sep)])) > chunk_size):
                window.pop(0)
        window.append((piece, sep))
    if window:
        chunks.append(join(window).strip())
    return [c for c in chunks if c]


def overlap_between(a: str, b: str) -> str:
    """Longest end of a that is also the start of b (the shared overlap)."""
    for n in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:n]):
            return b[:n]
    return ""


@dataclass
class Chunk:
    text: str
    page: int
    title: str
    keywords: list[str]
    hasCode: bool
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)


def get_splitter(chunk_size=500, chunk_overlap=150):
    """Use LangChain's real splitter (as in load_data.py) when installed,
    otherwise the small built-in copy above (identical at 500/150)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=min(chunk_overlap, chunk_size - 1))
        return s.split_text
    except Exception:
        return lambda text: split_text(text, chunk_size, chunk_overlap)


def chunk_pages(pages: list[dict], chunk_size=500, chunk_overlap=150) -> list[Chunk]:
    out: list[Chunk] = []
    splitter = get_splitter(chunk_size, chunk_overlap)
    for p in pages:
        for t in splitter(p["text"]):
            out.append(Chunk(t, p["page"], p["title"], p["keywords"], p["hasCode"], len(out)))
    return out


# ---------------------------------------------------------------------------
# 4. Embed: a keyword (TF-IDF) vector stands in for Voyage AI's 1,024 numbers.
#    A small synonym list imitates the way real embeddings capture meaning,
#    for example knowing that "table" and "collection" are related.
# ---------------------------------------------------------------------------

STOPWORDS = set(
    """a an the and or but if of to in on at by for with from as is are was were be been
    it its this that these those you your we our they their there here what which who
    whom how when where why do does did can could would should will shall may might
    into than then so not no yes about over under up down out more most some any all
    each one two just also much very lot like such use used using get got have has had
    i me my he she him her them us am between make makes made whole same thing things
    tell explain describe""".split()
)

SYNONYMS = {
    "table": "collection", "tables": "collection", "row": "document", "rows": "document",
    "record": "document", "records": "document", "column": "field", "columns": "field",
    "acid": "transaction", "atomic": "transaction", "atomicity": "transaction",
    "commit": "transaction", "rollback": "transaction",
    "join": "lookup", "joins": "lookup", "relationship": "lookup",
    "search": "find", "query": "find", "queries": "find", "filter": "find", "select": "find",
    "modify": "update", "change": "update", "edit": "update",
    "speed": "performance", "fast": "performance", "slow": "performance", "faster": "performance",
    "scale": "sharding", "scaling": "sharding", "partition": "sharding",
    "backup": "replica", "failover": "replica", "availability": "replica", "copies": "replica",
    "group": "aggregation", "summarize": "aggregation", "sum": "aggregation", "total": "aggregation",
    "location": "geospatial", "map": "geospatial", "semantic": "vector", "embedding": "vector",
    "embeddings": "vector", "ai": "vector", "rag": "vector", "began": "added", "begin": "added",
    "start": "added", "introduced": "added", "support": "added", "supporting": "added",
    "difference": "versus", "differ": "versus",
}


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\$?[a-zA-Z][a-zA-Z0-9]*", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        w = SYNONYMS.get(w, w)
        out.append(_stem(w))
    return out


class KeywordEmbedder:
    """Fit once on all chunks, then turn any text into a unit-length vector."""

    def __init__(self, texts: list[str]):
        docs = [set(tokenize(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        self.vocab = sorted(df)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        n = max(len(texts), 1)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in self.vocab])

    @property
    def dims(self) -> int:
        return len(self.vocab)

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(tokenize(text)).items():
            if w in self.index:
                v[self.index[w]] = c
        v = v * self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def top_terms(self, vec: np.ndarray, n: int = 8) -> list[tuple[str, float]]:
        idx = np.argsort(-vec)[:n]
        return [(self.vocab[i], float(vec[i])) for i in idx if vec[i] > 0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def project_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squash vectors to 2-D (PCA via SVD) so people can see 'nearness'."""
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    comps = vt[:2]
    return (matrix - mean) @ comps.T, mean, comps


# ---------------------------------------------------------------------------
# 5. Retrieve: what $vectorSearch does, in miniature
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    chunk: Chunk
    score: float


def search(query: str, chunks: list[Chunk], embedder: KeywordEmbedder,
           vectors: np.ndarray, k: int = 3, pre_filter_no_code: bool = False,
           score_threshold: float = 0.0) -> list[Hit]:
    q = embedder.embed(query)
    hits = []
    for c, v in zip(chunks, vectors):
        if pre_filter_no_code and c.hasCode:
            continue  # pre-filter: never even scored
        hits.append(Hit(c, cosine(q, v)))
    hits.sort(key=lambda h: -h.score)
    hits = hits[:k]
    return [h for h in hits if h.score >= score_threshold and h.score > 0]


# ---------------------------------------------------------------------------
# 6. Generate: the prompt template from rag.py, and a keyless stand-in answer
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Do not answer the question if there is no given context.
Do not answer the question if it is not related to the context.
Context:
{context}
Question: {question}"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(h.chunk.text for h in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def extractive_answer(question: str, hits: list[Hit], min_score: float = 0.08) -> str:
    """No LLM: pick the sentences from the retrieved chunks that best match."""
    if not hits or hits[0].score < min_score:
        return "I don't know."
    q = set(tokenize(question))
    sentences = []
    for rank, h in enumerate(hits):
        for s in re.split(r"(?<=[.!?])\s+", h.chunk.text.replace("\n", " ")):
            s = s.strip()
            if len(s) < 25:
                continue
            overlap = len(q & set(tokenize(s)))
            sentences.append((overlap + h.score - rank * 0.01, s))
    sentences.sort(key=lambda x: -x[0])
    best = [s for score, s in sentences[:2] if score > 0.5]
    return " ".join(best) if best else "I don't know."


# ---------------------------------------------------------------------------
# 7. Optional live mode (only when the app owner adds keys to st.secrets)
# ---------------------------------------------------------------------------

def claude_answer(prompt: str, api_key: str, model: str = "claude-opus-5-5") -> str:
    """Same call shape as ask_claude() in rag.py."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {}
    if model in {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}:
        # Retry on another model if a request is declined (same as rag.py)
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": "low"},  # short factual answers, keep it quick
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        return "I can't answer that."
    return "".join(b.text for b in response.content if b.type == "text")


def atlas_search(query: str, mongodb_uri: str, voyage_key: str, k: int = 3,
                 pre_filter_no_code: bool = False,
                 db: str = "book_mongodb_chunks", coll: str = "chunked_data",
                 index: str = "vector_index") -> list[dict]:
    """Real retrieval against the Atlas collection that load_data.py filled."""
    import certifi
    import voyageai
    from pymongo import MongoClient

    qvec = voyageai.Client(api_key=voyage_key).embed(
        [query], model="voyage-4", input_type="query"
    ).embeddings[0]
    stage = {
        "index": index,
        "path": "embedding",
        "queryVector": qvec,
        "numCandidates": k * 20,
        "limit": k,
    }
    if pre_filter_no_code:
        stage["filter"] = {"hasCode": {"$eq": False}}
    client = MongoClient(mongodb_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {"_id": 0, "text": 1, "title": 1, "page": 1, "hasCode": 1,
                      "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(client[db][coll].aggregate(pipeline))
