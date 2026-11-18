"""
RAG with MongoDB - an interactive explainer.

Run locally:   streamlit run app.py
Deploy:        see README.md (GitHub + Streamlit Community Cloud)

Works with NO keys (demo mode). If the owner adds keys in Streamlit secrets,
a "live" switch lets visitors send questions to Claude and to the real Atlas
collection built by load_data.py.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import rag_engine as rag

st.set_page_config(
    page_title="RAG with MongoDB",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="auto",
)

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Look and feel: MongoDB presentation palette
# ---------------------------------------------------------------------------
SLATE, FOREST, SPRING = "#001E2B", "#00684A", "#00ED64"
LAVENDER, PURPLE, LIME, MIST = "#F4ECFD", "#B45AF2", "#E9FF99", "#E3FCF7"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Public+Sans:wght@400;500;600;700&family=Source+Code+Pro:wght@400;600&display=swap');
:root {{ --slate:{SLATE}; --forest:{FOREST}; --spring:{SPRING}; --lav:{LAVENDER};
        --purple:{PURPLE}; --lime:{LIME}; --mist:{MIST}; --ink:#1C2D38; --muted:#5C6C75; --line:#E1E7E5; }}
html, body, [class*="css"], .stMarkdown, p, li, label {{ font-family:'Public Sans', sans-serif; }}
.stApp {{ background:#FFFFFF; color:var(--ink); }}
h1, h2, h3 {{ font-family:'Source Serif 4', Georgia, serif !important; font-weight:400 !important; color:var(--slate); letter-spacing:-0.01em; }}
code, pre {{ font-family:'Source Code Pro', monospace !important; }}
.block-container {{ padding-top:1.6rem; max-width:1180px; }}

/* Sidebar like the dark title slides */
[data-testid="stSidebar"] {{ background:var(--slate); }}
[data-testid="stSidebar"] * {{ color:#E8F0EC !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:3px 6px; border-radius:8px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:rgba(0,237,100,.10); }}
[data-testid="stSidebar"] input {{ color:var(--ink) !important; }}
[data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,.15); }}

/* Hero band */
.hero {{ position:relative; overflow:hidden; border-radius:22px; padding:34px 38px 30px; margin-bottom:22px; }}
.hero.dark {{ background:var(--slate); color:#fff; }}
.hero.light {{ background:#fff; border:1px solid var(--line); }}
.hero .blob {{ position:absolute; right:-70px; bottom:-90px; width:300px; height:240px; background:var(--forest); border-radius:140px 140px 0 140px; }}

.hero .arc {{ position:absolute; right:60px; top:-120px; width:300px; height:300px; border:1.5px solid var(--spring); border-radius:50%; opacity:.8; }}
.hero.light .arc {{ right:-150px; top:-200px; }}
.hero .kicker {{ display:inline-block; font:600 11px 'Source Code Pro', monospace; letter-spacing:.12em; text-transform:uppercase;
               background:var(--forest); color:#fff; padding:5px 14px; border-radius:999px; margin-bottom:14px; }}
.hero.light .kicker {{ background:var(--mist); color:var(--forest); }}
.hero h1 {{ font-size:44px; line-height:1.08; margin:0 0 10px; color:inherit !important; position:relative; }}
.hero.dark h1 {{ color:#fff !important; }}
.hero p {{ font-size:18px; max-width:640px; margin:0; position:relative; opacity:.92; }}
.leaf {{ display:inline-block; width:12px; height:18px; background:var(--spring); border-radius:50% 0 50% 0; transform:rotate(45deg); margin-right:10px; }}

/* Why / What / When / Where cards */
.w4 {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; margin:4px 0 18px; }}
@media (max-width:760px) {{ .w4 {{ grid-template-columns:1fr; }} .hero h1 {{ font-size:32px; }} }}
.card {{ border-radius:16px; padding:16px 18px; border:1px solid var(--line); background:#fff; }}
.card .tag {{ font:600 11px 'Source Code Pro', monospace; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }}
.card .body {{ font-size:15.5px; line-height:1.5; }}
.card.why {{ background:var(--lav); border-color:#E6D6F8; }}  .card.why .tag {{ color:#7A2FB8; }}
.card.what {{ background:var(--mist); border-color:#C6F1E4; }} .card.what .tag {{ color:var(--forest); }}
.card.when {{ background:#FFF8E1; border-color:#F7E3A6; }}    .card.when .tag {{ color:#946200; }}
.card.where {{ background:#EEF4FF; border-color:#D3E1FB; }}   .card.where .tag {{ color:#1D4FB5; }}

.callout {{ border-radius:14px; padding:14px 18px; margin:10px 0 16px; font-size:15.5px; line-height:1.5; }}
.callout.analogy {{ background:var(--slate); color:#fff; }}
.callout.analogy b {{ color:var(--spring); }}
.callout.takeaway {{ background:var(--lime); color:var(--slate); border:1px solid #D6EE73; }}
.callout.example {{ background:#fff; border:1px dashed var(--forest); }}
.callout.warn {{ background:#FFF4F2; border:1px solid #F6C9BF; }}
.chip {{ display:inline-block; font:600 11px 'Source Code Pro', monospace; padding:3px 10px; border-radius:999px; margin:0 6px 6px 0;
        background:var(--mist); color:var(--forest); border:1px solid #C6F1E4; }}
.chip.dark {{ background:var(--slate); color:var(--spring); border-color:var(--slate); }}
.chip.purple {{ background:var(--lav); color:#7A2FB8; border-color:#E6D6F8; }}

/* Flow diagram */
.flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:6px 0 14px; }}
.node {{ border-radius:14px; padding:10px 14px; border:1.5px solid var(--line); background:#fff; min-width:96px; text-align:center;
        font-size:13.5px; transition:all .2s; }}
.node .ico {{ font-size:20px; display:block; }}
.node.on {{ background:var(--slate); color:#fff; border-color:var(--spring); box-shadow:0 0 0 3px rgba(0,237,100,.25); }}
.node.done {{ background:var(--mist); border-color:#9BE7C9; }}
.arrow {{ color:var(--forest); font-size:20px; }}
.lane {{ font:600 11px 'Source Code Pro', monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-top:6px; }}

.chunkbox {{ border-radius:12px; padding:10px 12px; margin:0 0 10px; font:13.5px/1.5 'Source Code Pro', monospace; white-space:pre-wrap; }}
.ov {{ background:#FFE58A; border-radius:3px; }}
.hit {{ border:1px solid var(--line); border-left:5px solid var(--forest); border-radius:12px; padding:12px 14px; margin-bottom:10px; background:#fff; }}
.hit .meta {{ font:12px 'Source Code Pro', monospace; color:var(--muted); margin-bottom:6px; }}
.hit .score {{ float:right; font:600 13px 'Source Code Pro', monospace; color:var(--forest); }}
.answer {{ background:var(--slate); color:#fff; border-radius:16px; padding:16px 20px; font-size:16.5px; line-height:1.55; }}
.answer .lbl {{ font:600 11px 'Source Code Pro', monospace; letter-spacing:.12em; color:var(--spring); text-transform:uppercase; display:block; margin-bottom:6px; }}
.small {{ font-size:13px; color:var(--muted); }}
div[data-testid="stMetricValue"] {{ font-family:'Source Serif 4', serif; color:var(--forest); }}
.stButton > button[kind="primary"] {{ background:var(--forest); border-color:var(--forest); }}
.stButton > button[kind="primary"]:hover {{ background:#00593F; border-color:#00593F; }}
.stTabs [aria-selected="true"] {{ color:var(--forest) !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    """HTML-escape and stop Streamlit from reading $ signs as math."""
    return html.escape(text).replace("$", "&#36;")


def show(markup: str) -> None:
    # Streamlit treats indented lines as code blocks, so flatten the HTML
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, dark: bool = True) -> None:
    deco = '<div class="arc"></div><div class="blob"></div>' if dark else '<div class="arc"></div>'
    show(f"""
    <div class="hero {'dark' if dark else 'light'}">{deco}
      <div class="kicker">{esc(kicker)}</div>
      <h1><span class="leaf"></span>{esc(title)}</h1>
      <p>{esc(subtitle)}</p>
    </div>""")


def w4(why: str, what: str, when: str, where: str) -> None:
    """The Why / What / When / Where block used on every concept page."""
    cards = [("why", "Why", why), ("what", "What", what), ("when", "When", when), ("where", "Where", where)]
    inner = "".join(
        f'<div class="card {c}"><div class="tag">{label}</div><div class="body">{body}</div></div>'
        for c, label, body in cards
    )
    show(f'<div class="w4">{inner}</div>')


def callout(kind: str, body: str, title: str | None = None) -> None:
    head = f"<b>{title}</b> " if title else ""
    show(f'<div class="callout {kind}">{head}{body}</div>')


def chips(items: list[str], style: str = "") -> str:
    return "".join(f'<span class="chip {style}">{esc(i)}</span>' for i in items)


def deck_ref(text: str) -> None:
    show(f'<div style="margin:-6px 0 12px">{chips(["Deck: " + text], "purple")}</div>')


# ---------------------------------------------------------------------------
# Secrets / live mode (all optional)
# ---------------------------------------------------------------------------

def secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:  # no secrets file at all: demo mode
        return None


ANTHROPIC_KEY = secret("ANTHROPIC_API_KEY")
MONGODB_URI = secret("MONGODB_URI")
VOYAGE_KEY = secret("VOYAGE_API_KEY")
CLAUDE_MODEL = secret("CLAUDE_MODEL") or "claude-opus-5-5"
ACCESS_CODE = secret("LIVE_ACCESS_CODE")


# ---------------------------------------------------------------------------
# Build the demo pipeline once (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def build_index(chunk_size: int = 500, chunk_overlap: int = 150):
    kept, dropped = rag.sanitize(rag.DEMO_PAGES)
    chunks = rag.chunk_pages(kept, chunk_size, chunk_overlap)
    embedder = rag.KeywordEmbedder([c.text for c in chunks])
    vectors = np.array([embedder.embed(c.text) for c in chunks])
    return kept, dropped, chunks, embedder, vectors


KEPT, DROPPED, CHUNKS, EMBEDDER, VECTORS = build_index()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = [
    "🏠  Start here",
    "1 · Why RAG?",
    "2 · The big picture",
    "3 · Load & sanitize",
    "4 · Chunking",
    "5 · Metadata",
    "6 · Embeddings",
    "7 · Store & index in Atlas",
    "8 · Retrieval",
    "9 · Generation",
    "10 · Try the whole pipeline",
    "11 · How I built my app",
    "12 · Recap & quiz",
]

if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]


def go(offset: int) -> None:
    i = PAGES.index(st.session_state.nav) + offset
    st.session_state.nav = PAGES[max(0, min(len(PAGES) - 1, i))]


def nav_buttons() -> None:
    st.write("")
    i = PAGES.index(st.session_state.nav)
    st.progress((i + 1) / len(PAGES), text=f"Step {i} of {len(PAGES) - 1}" if i else "Welcome")
    left, _, right = st.columns([1, 2, 1])
    if i > 0:
        left.button(f"← {PAGES[i - 1].split('·')[-1].strip()}", on_click=go, args=(-1,), width="stretch")
    if i < len(PAGES) - 1:
        right.button(f"{PAGES[i + 1].split('·')[-1].strip()} →", on_click=go, args=(1,),
                     type="primary", width="stretch")


with st.sidebar:
    show('<div style="font:600 20px \'Source Serif 4\',serif;margin:4px 0 2px"><span class="leaf"></span>RAG with MongoDB</div>'
         '<div class="small" style="color:#9FB3AC!important;margin-bottom:10px">An interactive explainer</div>')
    st.radio("Sections", PAGES, key="nav", label_visibility="collapsed")
    st.divider()

    live_ready = bool(ANTHROPIC_KEY)
    if live_ready:
        st.markdown("**Live mode** is available")
        unlocked = True
        if ACCESS_CODE:
            code = st.text_input("Access code", type="password", help="Ask the presenter for the code")
            unlocked = code == ACCESS_CODE
        st.session_state.live = st.toggle("Use real Claude answers", value=False, disabled=not unlocked)
        st.session_state.atlas = bool(MONGODB_URI and VOYAGE_KEY) and st.toggle(
            "Search my real Atlas collection", value=False, disabled=not unlocked)
    else:
        st.session_state.live = False
        st.session_state.atlas = False
        st.markdown("**Demo mode**: everything runs in your browser session, no keys needed.")

    with st.expander("Glossary"):
        st.markdown(
            "- **LLM**: large language model, e.g. Claude\n"
            "- **Chunk**: a small piece of a document\n"
            "- **Embedding**: a list of numbers that captures meaning\n"
            "- **Vector search**: find chunks whose numbers are closest to the question's\n"
            "- **Context**: the chunks we paste into the prompt\n"
            "- **Grounded**: the answer comes from the context, not from memory"
        )


page = st.session_state.nav

# ===========================================================================
# PAGES
# ===========================================================================

if page == PAGES[0]:
    hero("Welcome", "RAG with MongoDB",
         "How to make an AI answer questions from YOUR documents, accurately, with sources. "
         "Explained one idea at a time, with a working demo, and the story of how I built my own app.")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("### The one-sentence version")
        callout("analogy",
                "RAG turns a <b>closed-book exam</b> into an <b>open-book exam</b>. "
                "Before the AI answers, we look up the right pages in our own library "
                "(MongoDB Atlas) and hand them over with the question. "
                "The AI then answers <i>from those pages</i>, instead of from memory.")
        st.markdown("### What you will be able to do")
        st.markdown(
            "1. **Explain** why plain LLMs make things up, and how RAG fixes it\n"
            "2. **Describe** the two halves of a RAG app: *preparing the data* and *answering questions*\n"
            "3. **Choose** a chunking strategy and understand embeddings\n"
            "4. **Build** one with MongoDB Atlas Vector Search, LangChain, Voyage AI and Claude"
        )
    with c2:
        st.markdown("### How to use this app")
        callout("example",
                "Each step follows the same pattern:<br><br>"
                "🟣 <b>Why</b> does this step exist?<br>"
                "🟢 <b>What</b> is it, in plain words?<br>"
                "🟡 <b>When</b> do you need it?<br>"
                "🔵 <b>Where</b> does it live in the app?<br><br>"
                "Then a <b>live demo</b> you can play with, the <b>code</b> from my app, "
                "and a <b>key takeaway</b> in green.")
        st.markdown("### The example we use throughout")
        st.markdown(
            "An app that answers questions about MongoDB, using **The Little MongoDB Book** as "
            "its only source. Ask it about MongoDB and it answers with sources; ask it *“Why is the sky blue?”* "
            "and it says *“I don't know.”*"
        )

    st.markdown("### The journey")
    steps = ["Why RAG", "Big picture", "Load", "Chunk", "Metadata", "Embed", "Store", "Retrieve", "Generate", "Try it", "My build", "Quiz"]
    show('<div class="flow">' + '<span class="arrow">›</span>'.join(
        f'<div class="node">{s}</div>' for s in steps) + "</div>")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[1]:
    hero("Step 1 · The problem", "Why do we need RAG?",
         "LLMs are brilliant writers with three blind spots. RAG fixes all three.", dark=False)
    deck_ref("slides 4–10")

    w4(why="LLMs <b>hallucinate</b>: they give confident, wrong answers when they lack the facts. "
           "In a business app, one wrong answer can cost trust.",
       what="<b>Retrieval-Augmented Generation</b>: first <i>retrieve</i> relevant facts from your data, "
            "then <i>augment</i> the prompt with them, then let the LLM <i>generate</i> the answer.",
       when="Whenever answers must come from <b>private</b>, <b>recent</b> or <b>large</b> sources: "
            "company docs, product manuals, policies, support tickets, a book.",
       where="Between your user and the LLM. Your app searches MongoDB Atlas first, then calls Claude "
             "with the question <i>plus</i> the search results.")

    st.markdown("### The three blind spots")
    a, b, c = st.columns(3)
    with a:
        callout("example", "<b>1. Trained on public data</b><br>It never saw your internal docs, so it "
                "can only guess about them.")
    with b:
        callout("example", "<b>2. Training data is stale</b><br>It knows the world up to a cut-off date. "
                "Last week's changes are invisible.")
    with c:
        callout("example", "<b>3. Limited context window</b><br>It can only read so much at once. "
                "Pasting a whole library is slow, costly and confusing.")

    st.markdown("### See the difference")
    st.caption("Pick a question. The left side shows the typical behaviour of an LLM on its own (illustrative); "
               "the right side runs this demo's real retrieval.")
    questions = {
        "When did MongoDB begin supporting multi-document transactions?": (
            "Probably answers from memory. It might be right, but you can't tell, and it can't show a source. "
            "If it's wrong, it sounds just as confident.", "public"),
        "What does our team's runbook say about failover?": (
            "It has never seen your runbook. It will either refuse or, worse, invent a plausible-sounding procedure.",
            "private"),
        "What did our team ship last Tuesday?": (
            "Its training stopped months ago. It cannot know, but may still produce an answer.", "stale"),
    }
    q = st.radio("Question", list(questions), label_visibility="collapsed")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🤖 LLM alone")
        callout("warn", questions[q][0])
    with right:
        st.markdown("#### 🍃 LLM + RAG")
        hits = rag.search(q, CHUNKS, EMBEDDER, VECTORS, k=3, score_threshold=0.15)
        if hits:
            ans = rag.extractive_answer(q, hits)
            show(f'<div class="answer"><span class="lbl">Grounded answer</span>{esc(ans)}</div>')
            st.caption(f"Source: page {hits[0].chunk.page}, “{hits[0].chunk.title}” (similarity {hits[0].score:.2f})")
        else:
            show('<div class="answer"><span class="lbl">Grounded answer</span>I don\'t know.</div>')
            st.caption("Nothing relevant in the library, so it refuses instead of guessing. "
                       "Load the runbook or release notes into MongoDB and this question becomes answerable.")

    st.markdown("### Why not paste the whole book into the prompt?")
    whole = sum(len(p["text"]) for p in KEPT)
    sent = sum(len(h.chunk.text) for h in rag.search("transactions", CHUNKS, EMBEDDER, VECTORS, 3))
    m1, m2, m3 = st.columns(3)
    m1.metric("Whole demo book", f"{whole:,} chars")
    m2.metric("What RAG sends (top 3 chunks)", f"{sent:,} chars")
    m3.metric("Saved", f"{100 - 100 * sent / whole:.0f}%")
    st.caption("The real Little MongoDB Book is about 30× larger than this demo, and company knowledge bases are "
               "millions of times larger. Sending only the relevant pieces is faster, cheaper and more accurate.")

    callout("takeaway", "LLMs are great at <b>language</b>, not at <b>knowing your facts</b>. "
            "RAG supplies the facts at question time, so answers are current, private-data aware and traceable.",
            "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[2]:
    hero("Step 2 · Architecture", "The big picture",
         "Every RAG app has two halves: prepare the library once, then answer questions all day.", dark=False)
    deck_ref("slides 11–24")

    w4(why="Separating the slow work (reading and indexing documents) from the fast work (answering) "
           "makes answers quick and cheap.",
       what="<b>Ingestion</b> (load_data.py) turns documents into searchable chunks in MongoDB. "
            "<b>Querying</b> (rag.py) finds the right chunks and asks the LLM.",
       when="Ingestion runs when documents change (once, nightly, or on a trigger). "
            "Querying runs every time someone asks a question.",
       where="Both halves meet in one MongoDB Atlas collection: "
             "<code>book_mongodb_chunks.chunked_data</code>, with a vector index on it.")

    ingest = [("📄", "Load", "Read the PDF into pages"), ("🧹", "Sanitize", "Drop empty or junk pages"),
              ("🏷️", "Tag", "Claude adds title, keywords, hasCode"), ("✂️", "Chunk", "Split into 500-char pieces"),
              ("🔢", "Embed", "Voyage AI turns text into 1,024 numbers"), ("🍃", "Store", "Save chunks + vectors in Atlas")]
    query = [("❓", "Question", "User asks something"), ("🔢", "Embed", "Same model turns the question into numbers"),
             ("🔎", "Vector search", "Atlas finds the 3 closest chunks"), ("📝", "Prompt", "Chunks + question go into a template"),
             ("🤖", "Claude", "Writes an answer from the context only"), ("✅", "Answer", "Grounded, or “I don't know”")]
    all_steps = [("Ingestion", s) for s in ingest] + [("Query", s) for s in query]

    step = st.slider("Walk through the pipeline step by step", 1, len(all_steps), 1,
                     help="Drag to move through all 12 steps")

    def lane(name, items, offset):
        nodes = []
        for j, (ico, title, _) in enumerate(items):
            n = offset + j + 1
            cls = "on" if n == step else ("done" if n < step else "")
            nodes.append(f'<div class="node {cls}"><span class="ico">{ico}</span>{esc(title)}</div>')
        return f'<div class="lane">{name}</div><div class="flow">' + '<span class="arrow">›</span>'.join(nodes) + "</div>"

    show(lane("① Ingestion · load_data.py · run once", ingest, 0))
    show(lane("② Query · rag.py · every question", query, len(ingest)))

    phase, (ico, title, desc) = all_steps[step - 1]
    examples = {
        "Load": "PyPDFLoader reads <code>mongodb.pdf</code> into one Document per page.",
        "Sanitize": "The blank cover page has 0 words, so it is dropped.",
        "Tag": "Page 34 gets <code>{title: 'Transactions', hasCode: false, keywords: ['ACID', ...]}</code>.",
        "Chunk": "Page 34 becomes two chunks that share 150 characters of overlap.",
        "Embed": "“MongoDB 4.0 added multi-document ACID transactions…” → [0.021, -0.113, 0.087, … 1,024 numbers]",
        "Store": "Each chunk is one MongoDB document: text + embedding + metadata.",
        "Question": "“When did MongoDB begin supporting multi-document transactions?”",
        "Vector search": "<code>$vectorSearch</code> returns the Transactions chunks with the highest scores.",
        "Prompt": "“Use the following pieces of context to answer the question… Context: … Question: …”",
        "Claude": "Reads only the context and writes a short answer.",
        "Answer": "“Multi-document transactions arrived in MongoDB 4.0 (2018).”",
    }
    ex = examples.get(title, "The question “When did…transactions?” becomes a 1,024-number vector, using the same Voyage model as ingestion.")
    callout("example", f"<b>{phase} · {ico} {esc(title)}</b>: {esc(desc)}.<br><span class='small'>Example: </span>{ex}")

    callout("analogy", "Think of a <b>library</b>. Ingestion is the librarian cataloguing every page by <i>topic</i>. "
            "Querying is a visitor asking a question: the librarian pulls the three most relevant pages and hands them "
            "to an expert (Claude), who reads them and answers.")
    callout("takeaway", "Two halves, one collection. <b>Ingestion</b> fills MongoDB with chunks and vectors; "
            "<b>querying</b> searches those vectors and lets the LLM answer from what it found.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[3]:
    hero("Step 3 · Ingestion", "Load & sanitize the data",
         "Garbage in, garbage out. Read the source, then throw away what will only confuse the search.", dark=False)
    deck_ref("slides 22–36")

    w4(why="Blank pages, cover pages and boilerplate create useless chunks that crowd out real answers. "
           "Sensitive data should never reach the index.",
       what="<b>Loading</b> reads files into text + metadata. <b>Sanitizing</b> removes empty, irrelevant or "
            "sensitive content and converts everything to one format (e.g. Markdown).",
       when="Always, as the first step of ingestion. Repeat whenever sources change or new formats arrive "
            "(PDF, HTML, images, spreadsheets).",
       where="Top of <code>load_data.py</code>: <code>PyPDFLoader</code> loads, a simple word-count rule sanitizes.")

    st.markdown("### Try it: what does the sanitizer keep?")
    min_words = st.slider("Keep pages with more than N words", 0, 120, 20,
                          help="My app uses 20, the same as the workshop")
    kept, dropped = rag.sanitize(rag.DEMO_PAGES, min_words)
    rows = [{"page": p["page"], "title": p["title"], "words": rag.word_count(p["text"]),
             "kept?": "✅ kept" if p in kept else "🗑️ dropped",
             "first words": (p["text"][:70] + "…") if p["text"] else "(empty)"} for p in rag.DEMO_PAGES]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    m1, m2 = st.columns(2)
    m1.metric("Pages kept", len(kept))
    m2.metric("Pages dropped", len(dropped))
    if min_words > 90:
        callout("warn", "Too strict! Real content pages are being thrown away. Sanitizing is a balance.")

    with st.expander("See the code from my app (load_data.py)", expanded=True):
        st.code('''# 1. Load
loader = PyPDFLoader("./sample_files/mongodb.pdf")
pages = loader.load()

# 2. Sanitize: keep pages with real text
cleaned_pages = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)
print(f"Loaded {len(pages)} pages, kept {len(cleaned_pages)}")''', language="python")
        st.markdown("Each loaded page looks like this:")
        st.code("page_content='' metadata={'source': './sample_files/mongodb.pdf', 'page': 0}", language="text")

    callout("takeaway", "Clean data beats clever code. Load everything into one format, then remove what "
            "won't help answer a question.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[4]:
    hero("Step 4 · Ingestion", "Chunking",
         "Break long text into small, focused pieces so search can find exactly the right paragraph.", dark=False)
    deck_ref("slides 37–42")

    w4(why="Search works best on focused pieces. A whole page mixes topics; a giant chunk wastes the context "
           "window. Small chunks make indexing efficient and answers precise.",
       what="Splitting text into pieces of a set size. <b>Overlap</b> repeats the end of one chunk at the start "
            "of the next so no sentence loses its meaning at a cut.",
       when="After sanitizing and before embedding. Re-chunk if answers miss context (chunks too small) or "
            "pull in noise (chunks too big).",
       where="<code>RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)</code> in "
             "<code>load_data.py</code>. It tries paragraphs first, then lines, then words.")

    st.markdown("### Common ways to chunk")
    strat = pd.DataFrame([
        ["Sentence", "One sentence per chunk", "FAQs, short facts"],
        ["Paragraph", "Split on blank lines", "Articles, documentation"],
        ["Page", "One page per chunk", "Slides, forms"],
        ["Section", "Split on headings", "Manuals, books with structure"],
        ["Semantic", "Split where the topic changes (uses embeddings)", "Long mixed content"],
        ["Fixed size + overlap ⭐", "N characters, with overlap (what my app uses)", "A solid default for anything"],
    ], columns=["Method", "How it splits", "Good for"])
    st.dataframe(strat, hide_index=True, width="stretch")

    st.markdown("### Try it: chunk a page yourself")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        choice = st.selectbox("Page", [p for p in KEPT], format_func=lambda p: f"p.{p['page']} · {p['title']}",
                              index=4)
    with c2:
        size = st.slider("chunk_size", 100, 800, 250, 50)
    with c3:
        overlap = st.slider("chunk_overlap", 0, 200, 60, 10)
    if overlap >= size:
        st.warning("Overlap must be smaller than the chunk size.")
    pieces = rag.get_splitter(size, overlap)(choice["text"])
    st.caption(f"{len(choice['text'])} characters → **{len(pieces)} chunks**. "
               "Yellow highlights show the overlap repeated from the previous chunk.")
    colors = ["#E3FCF7", "#F4ECFD", "#EEF4FF", "#FFF8E1"]
    for i, p in enumerate(pieces):
        ov = rag.overlap_between(pieces[i - 1], p) if i else ""
        body = (f'<span class="ov">{esc(ov)}</span>' + esc(p[len(ov):])) if ov else esc(p)
        show(f'<div class="chunkbox" style="background:{colors[i % 4]}"><b>chunk {i + 1}</b> · {len(p)} chars<br>{body}</div>')

    callout("analogy", "Chunking is like cutting a long film into <b>scenes</b>. Too long and you can't find the "
            "moment you want; too short and a scene makes no sense on its own. Overlap is the "
            "“previously on…” recap at the start of each scene.")
    with st.expander("See the code from my app"):
        st.code('''text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)   # each chunk keeps its page's metadata''', language="python")
    callout("takeaway", "Chunk size is a dial between <b>precision</b> (small) and <b>context</b> (large). "
            "500 characters with 150 overlap is a sensible starting point, so experiment from there.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[5]:
    hero("Step 5 · Ingestion", "Metadata: data about your data",
         "Labels like title, keywords and hasCode let you filter the search before it even starts.", dark=False)
    deck_ref("slides 43–47")

    w4(why="Vectors capture meaning, but not facts like “this page contains code”. Metadata lets you narrow "
           "the search space, which improves both speed and relevance.",
       what="Extra fields saved with every chunk. Some come free (source, page). Others are "
            "<b>generated by an LLM</b>: my app asks Claude for a <code>title</code>, <code>keywords</code> and <code>hasCode</code>.",
       when="During ingestion, before chunking, so every chunk inherits its page's tags. Add fields you will "
            "want to filter on later: date, product, language, department.",
       where="<code>tag_page()</code> in <code>load_data.py</code> uses Claude structured outputs with a Pydantic model; "
             "results are cached in <code>tagged_pages.json</code>.")

    st.markdown("### The tags on our demo pages")
    tag_df = pd.DataFrame([{"page": p["page"], "title": p["title"], "keywords": ", ".join(p["keywords"]),
                            "hasCode": "✅" if p["hasCode"] else "—"} for p in KEPT])
    st.dataframe(tag_df, hide_index=True, width="stretch")

    st.markdown("### Try it: one tagged chunk, as it is stored")
    c = st.selectbox("Chunk", CHUNKS, format_func=lambda c: f"chunk {c.chunk_id} · p.{c.page} · {c.title}")
    st.code(f"page_content={c.text[:160]!r}…\nmetadata={{'title': {c.title!r}, 'keywords': {c.keywords}, "
            f"'hasCode': {c.hasCode}, 'source': '{rag.SOURCE}', 'page': {c.page}}}", language="python")

    with st.expander("See the code from my app (Claude replaces the workshop's OpenAI tagger)", expanded=True):
        st.code('''class PageMetadata(BaseModel):
    title: str
    keywords: List[str]
    hasCode: bool

def tag_page(page):
    response = claude.beta.messages.parse(
        model="claude-opus-5-5",
        max_tokens=2000,
        output_config={"effort": "low"},          # a simple extraction job
        messages=[{"role": "user", "content":
            "Read this page from a book about MongoDB and extract: a short title, "
            "a list of keywords, and whether the page contains code (hasCode).\\n\\n"
            + page.page_content}],
        output_format=PageMetadata,
    )
    page.metadata.update(response.parsed_output.model_dump())
    return page''', language="python")
    callout("analogy", "Metadata is the <b>label on a moving box</b>. You could open every box to find the "
            "kitchen stuff, or you could read “KITCHEN” on the side and skip the rest.")
    callout("takeaway", "Tag at ingestion, filter at query time. Good metadata makes retrieval faster and more "
            "accurate, and an LLM can generate it for you.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[6]:
    hero("Step 6 · Ingestion", "Embeddings: meaning as numbers",
         "An embedding model turns text into a long list of numbers. Similar meaning → nearby numbers.", dark=False)
    deck_ref("slides 15–16, 48–49")

    w4(why="Keyword search misses synonyms: “table” won't match “collection”. Embeddings let us search by "
           "<b>meaning</b>, so the right chunk is found even when the words differ.",
       what="A vector: a list of numbers (my app: <b>1,024</b> from Voyage AI <code>voyage-4</code>; the workshop: 1,536 from OpenAI). "
            "Each chunk and each question gets one. Closeness is measured with <b>cosine similarity</b>.",
       when="Once per chunk at ingestion, and once per question at query time. "
            "<b>Both must use the same model</b>, or the numbers won't be comparable.",
       where="<code>VoyageAIEmbeddings</code> in both <code>load_data.py</code> and <code>rag.py</code>. "
             "The vectors are stored in the <code>embedding</code> field of each MongoDB document.")

    callout("warn", "<b>About this demo:</b> real embeddings need an API call. To stay key-free, this page uses "
            f"simple keyword vectors ({EMBEDDER.dims} numbers, one per word in the demo vocabulary) plus a small "
            "synonym list to imitate meaning. The idea, and the maths of closeness, are the same.")

    st.markdown("### Try it: how similar are two sentences?")
    c1, c2 = st.columns(2)
    s1 = c1.text_input("Sentence A", "How do I speed up queries?")
    s2 = c2.text_input("Sentence B", "Use an index to improve performance")
    v1, v2 = EMBEDDER.embed(s1), EMBEDDER.embed(s2)
    sim = rag.cosine(v1, v2)
    st.progress(max(0.0, min(1.0, sim)), text=f"Cosine similarity: **{sim:.2f}**  (1.0 = same meaning, 0 = unrelated)")
    st.caption("Notice: almost no words are shared, yet they score high. Try B = “Unicorns weigh 600 pounds” (unrelated), "
               "or A = “A collection holds many documents” with B = “A table holds many rows” (synonyms).")
    with st.expander("Peek inside the vectors"):
        cc1, cc2 = st.columns(2)
        cc1.write({w: round(x, 3) for w, x in EMBEDDER.top_terms(v1)})
        cc2.write({w: round(x, 3) for w, x in EMBEDDER.top_terms(v2)})
        st.caption("A real Voyage vector looks like [0.0213, -0.1132, 0.0871, …] with 1,024 numbers, none of which "
                   "map to a single word. Meaning is spread across all of them.")

    st.markdown("### The map of meaning")
    st.caption("Every dot is one chunk of our demo book, squashed from many dimensions down to 2 so we can see it. "
               "Type a question: the ★ shows where it lands. Search returns the dots nearest to it.")
    mq = st.text_input("Your question", "How do I combine data from two collections?")
    pts, mean, comps = rag.project_2d(VECTORS)
    qv = EMBEDDER.embed(mq)
    qpt = (qv - mean) @ comps.T
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "topic": [c.title for c in CHUNKS],
                       "text": [c.text[:90] + "…" for c in CHUNKS],
                       "similarity": [round(rag.cosine(qv, v), 2) for v in VECTORS]})
    palette = [FOREST, PURPLE, "#1D4FB5", "#946200", "#E0457B", "#00A0B0", "#5C6C75", "#7A2FB8", "#2E7D32"]
    dots = alt.Chart(df).mark_circle(size=220, opacity=0.85).encode(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("topic", scale=alt.Scale(range=palette), legend=alt.Legend(orient="bottom", columns=3)),
        tooltip=["topic", "similarity", "text"])
    star = alt.Chart(pd.DataFrame({"x": [qpt[0]], "y": [qpt[1]], "label": ["your question"]})).mark_point(
        shape="diamond", size=500, filled=True, color=SLATE).encode(x="x", y="y", tooltip=["label"])
    st.altair_chart((dots + star).properties(height=380), width="stretch")
    best = df.sort_values("similarity", ascending=False).head(3)
    st.markdown("Closest chunks: " + ", ".join(f"**{t}** ({s})" for t, s in zip(best.topic, best.similarity)))

    with st.expander("See the code from my app"):
        st.code('''embeddings = VoyageAIEmbeddings(
    model="voyage-4",                         # 1,024 dimensions
    voyage_api_key=key_param.voyage_api_key,
)''', language="python")
    callout("analogy", "Embeddings are <b>GPS coordinates for meaning</b>. “Collection” and “table” live in the "
            "same neighbourhood; “unicorn weight” lives across town. Search means “find my nearest neighbours”.")
    callout("takeaway", "Embeddings let you search by meaning, not spelling. Use the <b>same model</b> for chunks "
            "and questions, and match the index's <code>numDimensions</code> to it (1,024 for voyage-4).", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[7]:
    hero("Step 7 · Ingestion", "Store & index in MongoDB Atlas",
         "Your chunks, their vectors and their metadata live side by side in one document, in one database.", dark=False)
    deck_ref("slides 48–54")

    w4(why="Keeping vectors next to the original text and metadata means one query does the search, the filter "
           "and returns the content. No second database to sync.",
       what="A normal MongoDB collection, plus an <b>Atlas Vector Search index</b> that knows which field holds the "
            "vectors, how many dimensions they have and how to measure closeness.",
       when="Store at the end of ingestion. Create the index once (Atlas UI → Search & Vector Search). Keep chunks "
            "up to date with <b>Atlas Triggers</b> or a <b>cron job</b> when source documents change.",
       where="Database <code>book_mongodb_chunks</code>, collection <code>chunked_data</code>, index "
             "<code>vector_index</code>, written by <code>MongoDBAtlasVectorSearch</code> in <code>load_data.py</code>.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### One chunk = one document")
        c = CHUNKS[[i for i, x in enumerate(CHUNKS) if x.title == "Transactions"][0]]
        vec_preview = [round(float(x), 4) for x in np.random.default_rng(34).normal(0, 0.05, 6)]
        doc = {"_id": "ObjectId('665fad58dbd06e797baf43af')", "text": c.text[:150] + "…",
               "embedding": f"{vec_preview[:4]} … 1,024 numbers",
               "title": c.title, "keywords": c.keywords, "hasCode": c.hasCode,
               "source": rag.SOURCE, "page": c.page}
        st.json(doc)
    with c2:
        st.markdown("### The vector index definition")
        model = st.radio("Embedding model", ["Voyage AI voyage-4 (my app)", "OpenAI (workshop)"], horizontal=True)
        dims = 1024 if model.startswith("Voyage") else 1536
        st.code(json.dumps({"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
            {"type": "filter", "path": "hasCode"}]}, indent=2), language="json")
        st.caption("**path**: the field holding vectors. **numDimensions**: must equal the model's output size. "
                   "**similarity**: how closeness is measured. **filter**: fields you can pre-filter on.")

    callout("warn", "<b>The #1 setup mistake:</b> an index with <code>numDimensions: 1536</code> while storing "
            "1,024-number Voyage vectors (or the reverse). The search silently returns nothing. Match the number "
            "to your model.")
    with st.expander("See the code from my app"):
        st.code('''collection.delete_many({})   # start fresh so re-runs don't create duplicates

vectorStore = MongoDBAtlasVectorSearch(
    collection=collection, embedding=embeddings, index_name="vector_index"
)

BATCH = 20                                  # small batches respect Voyage rate limits
for start in range(0, len(split_docs), BATCH):
    vectorStore.add_documents(split_docs[start:start + BATCH])''', language="python")
    callout("analogy", "Atlas is the <b>library</b> and the vector index is its <b>card catalogue sorted by "
            "topic</b>. Without the catalogue the books are there, but nobody can find them quickly.")
    callout("takeaway", "Store text + vector + metadata together, create a vector index whose dimensions match "
            "your model, and keep it fresh with Triggers or a scheduled job.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[8]:
    hero("Step 8 · Query", "Retrieval",
         "The retriever turns the question into a vector and asks Atlas for the closest chunks.", dark=False)
    deck_ref("slides 52–69")

    w4(why="The LLM can only be as good as what you hand it. Retrieval decides which facts it sees, so this "
           "is where most quality is won or lost.",
       what="A <b>retriever</b> embeds the question and runs <code>$vectorSearch</code>. Three dials: "
            "<b>k</b> (how many chunks), a <b>pre-filter</b> on metadata, and a <b>score threshold</b> to drop weak matches.",
       when="On every question, before calling the LLM. Tune k, filters and threshold by testing real questions "
            "and checking what comes back. Experiment!",
       where="<code>vectorStore.as_retriever(...)</code> inside <code>query_data()</code> in <code>rag.py</code>.")

    st.markdown("### Try it: tune the retriever")
    q = st.text_input("Question", "When did MongoDB begin supporting multi-document transactions?")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("k: chunks to return", 1, 6, 3)
    no_code = c2.toggle("Pre-filter: hasCode = false", value=False,
                        help="Only search chunks without code. The filter runs BEFORE similarity scoring.")
    thr = c3.slider("Score threshold", 0.0, 0.6, 0.0, 0.05,
                    help="Drop chunks below this similarity")
    use_atlas = st.session_state.get("atlas")
    if use_atlas:
        try:
            with st.spinner("Searching your real Atlas collection…"):
                res = rag.atlas_search(q, MONGODB_URI, VOYAGE_KEY, k, no_code)
            res = [r for r in res if r.get("score", 0) >= thr]
            st.success(f"Live: {len(res)} chunks from Atlas")
            for r in res:
                show(f'<div class="hit"><span class="score">{r.get("score", 0):.3f}</span>'
                     f'<div class="meta">p.{r.get("page")} · {esc(str(r.get("title")))} · hasCode={r.get("hasCode")}</div>'
                     f'{esc(str(r.get("text", ""))[:400])}</div>')
        except Exception as e:
            st.error(f"Atlas search failed: {e}")
    else:
        hits = rag.search(q, CHUNKS, EMBEDDER, VECTORS, k, no_code, thr)
        if not hits:
            callout("warn", "No chunks passed the filters. With nothing to go on, the generator should say "
                    "<b>“I don't know”</b>, which is exactly what we want.")
        for h in hits:
            show(f'<div class="hit"><span class="score">{h.score:.3f}</span>'
                 f'<div class="meta">p.{h.chunk.page} · {esc(h.chunk.title)} · hasCode={h.chunk.hasCode}</div>'
                 f'{esc(h.chunk.text)}</div>')

    t1, t2 = st.tabs(["LangChain code (my app)", "What Atlas actually runs"])
    with t1:
        if thr > 0:
            kw = f'"k": {k},' + (' "pre_filter": {"hasCode": {"$eq": False}},' if no_code else "") + f' "score_threshold": {thr}'
            st.code(f'retriever = vectorStore.as_retriever(\n    search_type="similarity_score_threshold",\n'
                    f'    search_kwargs={{{kw}}},\n)', language="python")
        else:
            kw = f'"k": {k}' + (', "pre_filter": {"hasCode": {"$eq": False}}' if no_code else "")
            st.code(f'retriever = vectorStore.as_retriever(\n    search_type="similarity",\n'
                    f'    search_kwargs={{{kw}}},\n)', language="python")
        st.caption("This code updates as you move the dials above.")
    with t2:
        stage = {"index": "vector_index", "path": "embedding", "queryVector": "[…1,024 numbers for your question…]",
                 "numCandidates": k * 20, "limit": k}
        if no_code:
            stage["filter"] = {"hasCode": {"$eq": False}}
        st.code(json.dumps([{"$vectorSearch": stage},
                            {"$project": {"text": 1, "title": 1, "score": {"$meta": "vectorSearchScore"}}}], indent=2),
                language="json")
        st.caption("$vectorSearch is an aggregation stage, so you can add normal MongoDB stages after it.")

    callout("example", "<b>Experiment:</b> type <i>“How do I update a document?”</i> and flip the pre-filter on. "
            "The Updating page has code, so it disappears from the results. Pre-filters are powerful, so use them deliberately.")
    callout("takeaway", "Retrieval quality = answer quality. Use <b>k</b> for breadth, <b>pre-filters</b> to narrow "
            "the search space, and a <b>threshold</b> to keep weak matches out of the prompt.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[9]:
    hero("Step 9 · Query", "Generation",
         "The retrieved chunks and the question go into a prompt template. The LLM answers from that context only.", dark=False)
    deck_ref("slides 70–82")

    w4(why="Retrieval finds facts; people want an answer in plain language. The prompt also sets the rules: "
           "<b>use only this context</b>, and say “I don't know” otherwise.",
       what="A <b>generator</b> = a prompt template + a generative model. LangChain's pipe <code>|</code> chains "
            "retriever → prompt → LLM → parser into one <code>rag_chain</code>.",
       when="After retrieval, for every question. Tighten the prompt if the model drifts from the context.",
       where="<code>template</code>, <code>ask_claude()</code> and <code>rag_chain</code> in <code>rag.py</code>. "
             "Claude (<code>claude-opus-5-5</code>) takes the place of the workshop's OpenAI chat model.")

    st.markdown("### The chain, one pipe at a time")
    parts = [("retrieve", "question → top-3 chunks"), ("custom_rag_prompt", "fill the template"),
             ("llm", "Claude writes the answer"), ("response_parser", "keep just the text")]
    show('<div class="flow"><div class="node on">query</div><span class="arrow">|</span>' +
         '<span class="arrow">|</span>'.join(f'<div class="node"><b>{a}</b><br><span class="small">{b}</span></div>'
                                             for a, b in parts) + "</div>")

    st.markdown("### Try it")
    st.session_state.setdefault("gen_q", "What is the difference between a collection and database in MongoDB?")
    b1, b2, b3, _ = st.columns([1, 1, 1, 1])
    if b1.button("Collection vs database?"):
        st.session_state.gen_q = "What is the difference between a collection and database in MongoDB?"
    if b2.button("Why is the sky blue?"):
        st.session_state.gen_q = "Why is the sky blue?"
    if b3.button("Speed up queries?"):
        st.session_state.gen_q = "How do I make my queries faster?"
    q = st.text_input("Question", key="gen_q")
    hits = rag.search(q, CHUNKS, EMBEDDER, VECTORS, 3, score_threshold=0.05)
    prompt = rag.build_prompt(q, hits)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**The filled-in prompt sent to the LLM**")
        st.code(prompt, language="text")
    with c2:
        st.markdown("**The answer**")
        if st.session_state.get("live"):
            if st.button("Ask Claude", type="primary"):
                with st.spinner("Claude is reading the context…"):
                    try:
                        st.session_state.gen_answer = (q, rag.claude_answer(prompt, ANTHROPIC_KEY, CLAUDE_MODEL))
                    except Exception as e:
                        st.error(f"Claude call failed: {e}")
            if st.session_state.get("gen_answer", ("", ""))[0] == q:
                show(f'<div class="answer"><span class="lbl">Claude · live</span>{esc(st.session_state.gen_answer[1])}</div>')
        else:
            ans = rag.extractive_answer(q, hits)
            show(f'<div class="answer"><span class="lbl">Demo answer · no LLM</span>{esc(ans)}</div>')
            st.caption("In demo mode the answer is the best-matching sentences from the context. "
                       "A real LLM would rephrase them into a natural answer, following the same rules.")
        if not hits:
            st.caption("No context was found, so the prompt's rules say: don't answer. That's the deck's "
                       "“Why is the sky blue?” → “I don't know.” moment.")

    with st.expander("See the code from my app (rag.py)"):
        st.code('''def ask_claude(prompt_value):
    response = claude.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt_value.to_string()}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

retrieve = {
    "context": retriever | (lambda docs: "\\n\\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
rag_chain = retrieve | custom_rag_prompt | RunnableLambda(ask_claude) | StrOutputParser()
answer = rag_chain.invoke(query)''', language="python")
    callout("analogy", "The prompt is the <b>exam instructions</b>: “Answer using only the attached pages. "
            "If the answer isn't there, write ‘I don't know’.” A good student follows them; so does a good LLM.")
    callout("takeaway", "Generation = template + model. The template's rules are what make the answer "
            "<b>grounded</b>, and what make “I don't know” a feature, not a failure.", "Key takeaway:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[10]:
    hero("Step 10 · Playground", "Try the whole pipeline",
         "Ask anything about MongoDB and watch every stage of RAG happen, with timings.", dark=True)

    q = st.text_input("Ask a question about MongoDB", "How does MongoDB keep running if a server fails?")
    c1, c2 = st.columns(2)
    k = c1.slider("k", 1, 5, 3, key="pk")
    no_code = c2.toggle("Exclude chunks with code", key="pf")
    if st.button("Run RAG", type="primary") or q:
        t0 = time.perf_counter()
        qv = EMBEDDER.embed(q)
        t1 = time.perf_counter()
        hits = rag.search(q, CHUNKS, EMBEDDER, VECTORS, k, no_code, 0.05)
        t2 = time.perf_counter()
        prompt = rag.build_prompt(q, hits)
        with st.expander(f"① Embed the question · {1000 * (t1 - t0):.1f} ms", expanded=False):
            st.write({w: round(x, 3) for w, x in EMBEDDER.top_terms(qv)})
            st.caption(f"{EMBEDDER.dims}-number demo vector (Voyage would give 1,024).")
        with st.expander(f"② Vector search · {len(hits)} chunks · {1000 * (t2 - t1):.1f} ms", expanded=True):
            for h in hits:
                show(f'<div class="hit"><span class="score">{h.score:.3f}</span>'
                     f'<div class="meta">p.{h.chunk.page} · {esc(h.chunk.title)}</div>{esc(h.chunk.text[:300])}…</div>')
            if not hits:
                st.write("No relevant chunks.")
        with st.expander("③ Build the prompt"):
            st.code(prompt, language="text")
        st.markdown("**④ Generate**")
        if st.session_state.get("live") and hits:
            with st.spinner("Asking Claude…"):
                try:
                    ans = rag.claude_answer(prompt, ANTHROPIC_KEY, CLAUDE_MODEL)
                    show(f'<div class="answer"><span class="lbl">Claude · live</span>{esc(ans)}</div>')
                except Exception as e:
                    st.error(f"Claude call failed: {e}")
        else:
            show(f'<div class="answer"><span class="lbl">Demo answer</span>{esc(rag.extractive_answer(q, hits))}</div>')
        if hits:
            st.caption("Sources: " + ", ".join(sorted({f'p.{h.chunk.page} {h.chunk.title}' for h in hits})))

    st.markdown("#### Ideas to try")
    st.markdown("- *What is an upsert?*  \n- *How do I group and total documents?*  \n- *Can I store an address inside a customer?*  \n"
                "- *Who won the World Cup?* (watch it refuse)")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[11]:
    hero("Step 11 · Behind the scenes", "How I built my RAG app",
         "The workshop code, rebuilt with Claude and Voyage AI on MongoDB Atlas, and what I learned along the way.", dark=True)

    st.markdown("### My stack vs the workshop's")
    st.dataframe(pd.DataFrame([
        ["Language", "Python", "Python 3.14 on a Mac"],
        ["Load PDF", "PyPDFLoader", "PyPDFLoader (same)"],
        ["Chunking", "RecursiveCharacterTextSplitter 500/150", "Same"],
        ["Metadata tagging", "create_metadata_tagger + gpt-3.5-turbo", "Claude structured outputs + Pydantic"],
        ["Embeddings", "OpenAI, 1,536 dims", "Voyage AI voyage-4, 1,024 dims"],
        ["Vector store", "MongoDB Atlas Vector Search", "MongoDB Atlas Vector Search (same)"],
        ["Answer model", "ChatOpenAI", "Claude claude-opus-5-5 via RunnableLambda"],
        ["Orchestration", "LangChain LCEL pipes", "LangChain LCEL pipes (same)"],
    ], columns=["Piece", "Workshop", "My app"]), hide_index=True, width="stretch")
    st.caption("Why Claude + Voyage? Anthropic recommends Voyage AI for embeddings, since Claude itself doesn't make embeddings. "
               "Swapping models only changed a few lines: the retrieval half with MongoDB stayed identical.")

    st.markdown("### Build steps")
    steps = [
        ("Set up", "Created a free Atlas cluster, a database user, allowed my IP; got Anthropic and Voyage API keys; "
                   "put them in <code>key_param.py</code> (never committed to Git)."),
        ("Install", "<code>pip install -r requirements.txt</code>: pymongo, langchain, langchain-mongodb, langchain-voyageai, anthropic, pypdf."),
        ("Ingest", "Ran <code>python load_data.py</code>: load → sanitize → tag with Claude → chunk → embed → store."),
        ("Index", "In Atlas, created <code>vector_index</code> on <code>chunked_data</code> with numDimensions 1024, cosine, and a hasCode filter."),
        ("Ask", "Ran <code>python rag.py</code>: the collection vs database question got a grounded answer; the sky question got “I don't know”."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        show(f'<div class="hit"><div class="meta">STEP {i}</div><b>{t}</b>: {d}</div>')

    st.markdown("### Problems I hit, and the fixes")
    probs = [
        ("CERTIFICATE_VERIFY_FAILED connecting to Atlas",
         "New Python versions on macOS don't trust any certificates by default.",
         "Pass <code>tlsCAFile=certifi.where()</code> to MongoClient."),
        ("Voyage AI RateLimitError halfway through",
         "Free tier limits how many tokens you can embed per minute.",
         "Send chunks in batches of 20; on a rate limit, wait 60 s and retry the same batch."),
        ("Re-running tagging cost time and money",
         "Every run asked Claude to tag every page again.",
         "Save tags to <code>tagged_pages.json</code> and reuse them on later runs."),
        ("Duplicate chunks after re-runs",
         "Each run appended the whole book again.",
         "<code>collection.delete_many({})</code> before inserting."),
        ("Workshop code didn't import",
         "LangChain moved modules (<code>langchain.text_splitter</code> → <code>langchain_text_splitters</code>) and gpt-3.5-turbo-0613 was retired.",
         "Updated imports; replaced OpenAI with Claude and Voyage."),
        ("Search returned nothing",
         "The index must match the embedding size.",
         "<code>numDimensions: 1024</code> for voyage-4 (the workshop's 1536 is for OpenAI)."),
    ]
    for title, cause, fix in probs:
        with st.expander(f"⚠️  {title}"):
            show(f"<b>Why it happened:</b> {cause}<br><b>Fix:</b> {fix}")

    st.markdown("### The full code")
    tabs = st.tabs(["load_data.py", "rag.py", "key_param.example.py"])
    for tab, name in zip(tabs, ["load_data.py", "rag.py", "key_param.example.py"]):
        with tab:
            path = HERE / "reference_code" / name
            st.code(path.read_text() if path.exists() else "(file missing)", language="python")
    callout("takeaway", "Building it yourself is where it sticks. The concepts stay the same whichever models you "
            "pick; most of the real work is plumbing: keys, certificates, rate limits and matching dimensions.",
            "What I learned:")
    nav_buttons()

# ---------------------------------------------------------------------------
elif page == PAGES[12]:
    hero("Step 12 · Recap", "What to remember",
         "Five verbs, one database, and one rule. Then check yourself with a quick quiz.", dark=True)

    st.markdown("### Five verbs")
    show('<div class="flow">' + '<span class="arrow">›</span>'.join(
        f'<div class="node on" style="min-width:120px"><span class="ico">{i}</span><b>{v}</b><br>'
        f'<span class="small" style="color:#9FB3AC">{d}</span></div>'
        for i, v, d in [("📄", "Load", "& sanitize"), ("✂️", "Chunk", "& tag"), ("🔢", "Embed", "& store"),
                        ("🔎", "Retrieve", "with $vectorSearch"), ("🤖", "Generate", "from context only")]) + "</div>")
    callout("analogy", "<b>Open-book exam.</b> MongoDB Atlas is the book, vector search finds the right pages, "
            "and Claude writes the answer from them. No page found? <b>“I don't know.”</b>")

    a, b = st.columns(2)
    with a:
        st.markdown("### Cheat sheet")
        st.markdown(
            "| Concept | One line |\n|---|---|\n"
            "| Hallucination | Confident, wrong answer |\n"
            "| Chunking | Small pieces for precise search |\n"
            "| Overlap | Repeated text so ideas aren't cut |\n"
            "| Metadata | Labels you can filter on |\n"
            "| Embedding | Meaning as numbers |\n"
            "| numDimensions | Must match the model (1,024 / 1,536) |\n"
            "| Pre-filter | Narrow the search before scoring |\n"
            "| k | How many chunks to retrieve |\n"
            "| Prompt template | The rules: context only |")
    with b:
        st.markdown("### Badge objectives covered")
        st.markdown("- ✅ **Identify the architecture** of a RAG application → steps 1–2\n"
                    "- ✅ **Describe chunking strategies** → step 4\n"
                    "- ✅ **Create a RAG application** → steps 3–11")

    st.markdown("### Quick quiz")
    quiz = [
        ("What problem does RAG mainly solve?",
         ["LLMs are slow", "LLMs lack your private or recent facts and may hallucinate", "LLMs can't write code"], 1),
        ("Why do we add overlap between chunks?",
         ["To make the index bigger", "So ideas cut at a boundary keep their context", "Atlas requires it"], 1),
        ("My app uses voyage-4 embeddings. What numDimensions must the index use?", ["1536", "1024", "500"], 1),
        ("What does a pre-filter like hasCode = false do?",
         ["Removes chunks after the answer", "Narrows the search space before similarity scoring", "Deletes code from MongoDB"], 1),
        ("The question and the chunks must be embedded…",
         ["with the same model", "with different models for variety", "only the chunks need embedding"], 0),
        ("Someone asks “Why is the sky blue?”. What should a well-built RAG app about MongoDB say?",
         ["Explain Rayleigh scattering", "I don't know", "Return a random chunk"], 1),
    ]
    score = 0
    answered = 0
    for i, (qq, opts, correct) in enumerate(quiz):
        choice = st.radio(f"**{i + 1}. {qq}**", opts, index=None, key=f"quiz{i}")
        if choice is not None:
            answered += 1
            if opts.index(choice) == correct:
                score += 1
                st.success("Correct!")
            else:
                st.error(f"Not quite. Answer: {opts[correct]}")
    if answered == len(quiz):
        st.metric("Your score", f"{score} / {len(quiz)}")
        if score == len(quiz):
            st.balloons()
    callout("takeaway", "RAG = <b>retrieve</b> the right facts from MongoDB Atlas, <b>augment</b> the prompt with "
            "them, and let the LLM <b>generate</b> a grounded answer.", "One line to remember:")
    nav_buttons()

st.markdown("---")
st.caption("Built for learning. Demo pages are short summaries written in the spirit of The Little MongoDB Book by "
           "Karl Seguin (CC BY-NC 3.0). MongoDB and Atlas are trademarks of MongoDB, Inc.; this is an independent, "
           "unofficial explainer.")
