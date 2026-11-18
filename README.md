# RAG with MongoDB: an interactive explainer

A Streamlit web app that teaches **Retrieval-Augmented Generation (RAG) with MongoDB Atlas**, one idea at a time, and shows how my own RAG app (Claude + Voyage AI + MongoDB Atlas Vector Search + LangChain) was built.

- 12 short steps, each with **Why / What / When / Where**, a hands-on demo, the real code, and a key takeaway
- Styled after the MongoDB "RAG with MongoDB" workshop deck
- Runs for anyone, **no API keys needed** (demo mode)
- Optional **live mode**: real Claude answers and real Atlas vector search, switched on with Streamlit secrets

## What's in this folder

| File | What it does |
|---|---|
| `app.py` | The app: all pages, layout and styling |
| `rag_engine.py` | The demo RAG pipeline (load, sanitize, chunk, embed, search, answer) plus optional live calls |
| `reference_code/` | My original `load_data.py` and `rag.py`, shown inside the app |
| `requirements.txt` | Python packages Streamlit Cloud installs |
| `.streamlit/config.toml` | MongoDB-style colours |
| `secrets.toml.example` | Template for optional live-mode keys (never put real keys in GitHub) |
| `.gitignore` | Stops secret files from being uploaded |

> Files starting with a dot (`.gitignore`, `.streamlit`) are **hidden** on a Mac. In Finder press **Cmd + Shift + .** (full stop) to show them.

---

# Step-by-step: from these files to a public website

You will do three things: **(A)** create a GitHub repository, **(B)** put these files in it, **(C)** connect it to Streamlit Community Cloud. No command line needed. Allow about 20 minutes the first time.

## A. Create the GitHub repository

1. Go to **https://github.com** and sign in (or click **Sign up** and create a free account).
2. Click the **+** in the top-right corner → **New repository**.
3. Fill in:
   - **Repository name**: `rag-mongodb-explainer`
   - **Description**: `Interactive RAG with MongoDB explainer built with Streamlit`
   - Choose **Public** (Streamlit's free tier deploys public repos most easily)
   - Leave "Add a README" **unticked** (we already have one)
4. Click **Create repository**. Keep this page open.

## B. Put the files in the repository

Pick **one** option.

### Option 1 (recommended): GitHub Desktop

GitHub Desktop is a free app that gives you buttons instead of commands, and handles hidden files properly.

1. Download it from **https://desktop.github.com**, install, open it and sign in with your GitHub account.
2. On your new repository's page on github.com, click **Set up in Desktop** (or in GitHub Desktop: **File → Clone repository**, pick `rag-mongodb-explainer`).
3. Choose where to save it (for example `Documents/GitHub/rag-mongodb-explainer`) and click **Clone**.
4. Unzip `rag-streamlit-app.zip`. Open the unzipped folder, press **Cmd + Shift + .** to show hidden files, select **everything inside** (including `.gitignore` and `.streamlit`) and copy it into the cloned `rag-mongodb-explainer` folder.
5. Back in GitHub Desktop you'll see the files listed under **Changes**. Bottom-left, type a summary such as `First version of RAG explainer` and click **Commit to main**.
6. Click **Publish branch** (or **Push origin**) at the top. Refresh your repository on github.com: the files are there.

### Option 2: upload in the browser

1. On your repository page click **uploading an existing file** (or **Add file → Upload files**).
2. Unzip `rag-streamlit-app.zip`, press **Cmd + Shift + .** in Finder to show hidden files, then drag **everything inside** the folder onto the page. Folders such as `reference_code` and `.streamlit` keep their structure.
3. Type a commit message such as `First version` and click **Commit changes**.
4. Check that `.gitignore` and `.streamlit/config.toml` appear in the file list. If `.streamlit/config.toml` is missing: **Add file → Create new file**, type `.streamlit/config.toml` as the name (typing the `/` creates the folder), paste the contents of that file, and click **Commit changes**. The app still works without it; it just uses Streamlit's default colours.

## C. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and click **Continue with GitHub**. Approve the request to connect your GitHub account.
2. Click **Create app** (top right).
3. When asked "Do you already have an app?", choose **Yup, I have an app**.
4. Fill in:
   - **Repository**: `your-github-username/rag-mongodb-explainer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): pick something memorable, e.g. `rag-mongodb-explainer`
5. Click **Advanced settings**, set **Python version** to **3.12**, leave Secrets empty for now, click **Save**.
6. Click **Deploy**. The first build takes 2 to 5 minutes while packages install. Your app opens at `https://<your-app-url>.streamlit.app`.
7. Share that link. Anyone can open it, no sign-in needed.

## D. Making changes later (this is the version control part)

Every change you push to GitHub is saved as a **commit** you can go back to, and Streamlit redeploys automatically within a minute.

- **GitHub Desktop**: edit a file in the folder (any text editor, e.g. VS Code) → GitHub Desktop shows the change → write a summary → **Commit to main** → **Push origin**.
- **Browser**: open the file on github.com → click the pencil ✏️ → edit → **Commit changes**.
- **Undo a mistake**: GitHub Desktop → **History** tab → right-click the bad commit → **Revert changes in commit** → **Push origin**.

## E. Optional: turn on live mode

Demo mode is enough to teach everything. Live mode lets visitors get real Claude answers and search your real Atlas collection.

1. In Streamlit Cloud open your app's **⋮** menu (or your workspace list) → **Settings** → **Secrets**.
2. Paste the contents of `secrets.toml.example` with your real values and click **Save**. The app restarts.
3. **Always set `LIVE_ACCESS_CODE`.** The app is public, and every live answer is billed to your Anthropic and Voyage accounts. Visitors must type the code in the sidebar before live mode switches on.
4. For Atlas search, Streamlit's servers must be allowed to connect: Atlas → **Network Access** → **Add IP Address** → `0.0.0.0/0`. That allows connections from anywhere, so make sure the database user has a strong password and only **read** access to `book_mongodb_chunks`.
5. Live search expects the collection and index your `load_data.py` created: `book_mongodb_chunks.chunked_data`, index `vector_index`, vectors in `embedding` (1,024 dims, voyage-4), and a `hasCode` filter field for the pre-filter switch.

Never commit real keys. `.gitignore` already blocks `.streamlit/secrets.toml` and `key_param.py`.

## Run it on your own computer (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at http://localhost:8501.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Error installing requirements" on Streamlit Cloud | Open **Manage app** (bottom right) to read the log. Check `requirements.txt` was uploaded and Python is 3.12. |
| App looks unstyled / default red buttons | `.streamlit/config.toml` didn't upload; see step B2.4. |
| "File not found: app.py" | Main file path must be exactly `app.py`, and the files must be at the top level of the repo, not inside a sub-folder. |
| Live mode toggle not visible | Secrets aren't saved, or `ANTHROPIC_API_KEY` is missing. |
| Atlas search times out | Network Access doesn't allow `0.0.0.0/0`, or the URI/password is wrong. |
| App "went to sleep" | Free apps sleep after a period without visitors. Click **Yes, get this app back up** and wait ~30 seconds. |

---

Demo pages are short summaries written in the spirit of *The Little MongoDB Book* by Karl Seguin (CC BY-NC 3.0). MongoDB and Atlas are trademarks of MongoDB, Inc. This is an independent learning project.
