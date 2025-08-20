import os, time
from pathlib import Path
import fitz
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError
import faiss
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import HumanMessage
from pathlib import Path
import re
import random

extensions = [
    "Under router rip, set version to 2.",
    "Under router rip, enable no auto-summary.",
    "Under router rip, enable passive-interface on a random set of interfaces.",
    "Under router rip, redistribute connected routes.",
    "Under router rip, redistribute static routes.",
    "Under router rip, redistribute ospf 1 metric 2.",
    "Under router rip, redistribute eigrp 100 metric 2.",
    "Under router rip, apply offset-list IN or OUT with a random hop count between 1 and 16.",
    "Set RIP timers basic with random values: update (5–60s), invalid (15–180s), holddown (15–180s), flush (30–240s).",
    "Apply distribute-list IN or OUT using a random ACL number.",
    "Enable RIP on a random subset of interfaces only.",
    "Apply authentication on RIP-enabled interfaces using key-chain rip-auth.",
    "On some interfaces, apply ip rip receive version 2.",
    "On some interfaces, apply ip rip send version 2.",
    "On some interfaces, apply ip rip authentication mode md5.",
    "On some interfaces, apply ip rip authentication key-chain rip-auth.",
    "Configure neighbor statements under router rip with fake neighbors from the same subnet.",
    "Apply maximum-paths <N> under router rip, where N is a random number between 1 and 6.",
    "Under router rip, use distance <value> where value is between 1 and 255.",
    "Under router rip, filter routes with route-map applied via distribute-list."
]

# ─── 0) Boilerplate: load PDFs, chunk, embed, build FAISS (your existing code) ──
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("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"))
folder = Path(r"Z:\Thesis\AConfig Guides\RIP")
raw_texts = []
for pdf in folder.rglob("irr-15-s-book.pdf"):
    doc = fitz.open(pdf)
    text = "".join(page.get_text("text") for page in doc)  # faster C-based parser :contentReference[oaicite:3]{index=3}
    doc.close()
    raw_texts.append(text)

# 2️⃣ Chunk Splitting (tuned for fewer segments)
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
chunks = []
for t in raw_texts:
    chunks.extend(splitter.split_text(t))

# 3️⃣ Batching + Rate-Limited Embeddings with Back-Off
def embed_batch(texts: list[str], max_retries: int = 5):
    backoff = 1
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [datum.embedding for datum in response.data]
        except RateLimitError:
            print(f"Rate limit hit, retrying in {backoff}s…")  # 429 handling :contentReference[oaicite:4]{index=4}
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Maximum retries exceeded for embeddings")

def batch_embed_all(
    chunks: list[str],
    batch_size: int = 20,      # small enough to avoid TPM spikes :contentReference[oaicite:5]{index=5}
    max_workers: int = 3       # throttle concurrency
) -> list[list[float]]:
    embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(embed_batch, chunks[i : i + batch_size])
            for i in range(0, len(chunks), batch_size)
        ]
        for fut in as_completed(futures):
            embeddings.extend(fut.result())
    return embeddings

emb_model = OpenAIEmbeddings()  
embeddings = batch_embed_all(chunks)  
d = len(embeddings[0])
index = faiss.IndexFlatIP(d)
index.add(np.array(embeddings, dtype="float32"))
vs = FAISS.from_embeddings(
    text_embeddings=list(zip(chunks, embeddings)),
    embedding=emb_model,
    metadatas=[{"text": c} for c in chunks],
)

# ─── 1) Retrieve the top‐K snippets yourself ───────────────────────────────────
num_chunks = len(chunks)
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})
docs = retriever.invoke("RIP")
ospf_docs = [
    d for d in docs 
    if d.metadata.get("source") == "irr-15-s-book.pdf"
]
context_snippets = "\n\n".join(d.page_content for d in docs)

# ─── 2) Load your IOS template from file ──────────────────────────────────────
ios_template = Path(r"Z:\Thesis\ios_template.txt").read_text()

for i in range(1, 51):
    # 1) Sample 7 new extensions
    chosen = random.sample(extensions, k=10)
    ext_block = "\n".join(f"- {e}" for e in chosen)

# ─── 3) Build a single prompt with both context + template + instructions ─────
prompt = f"""
You are a network architect. 
**Use ONLY the following context** from the provided PDF manuals—do not rely on any other knowledge.
**Create 50 router configurations**
Output them one after the other
Port Interfaces must include the advertised network to make it realistic.
Each block MUST start with for example "hostname R1" "hostname R2" etc.
For each router, assign it a unique hostname (e.g. R1, R2…).  
Use only subnets in the 192.168.*.*/24 space for VLANs, Make sure to choose a random number from 1-254 to replace the "*" in the ip's
If the context lacks any required command, leave that section blank and write UNKNOWN.

=== EXTRA TASKS ===
{ext_block}

=== CONTEXT SNIPPETS ===
{context_snippets}

=== IOS TEMPLATE ===
{ios_template}

=== INSTRUCTIONS ===
Fill in or extend the attached C7200 Router IOS template above so that it configures:
- Add no shutdown to all interfaces
- RIP
- All networks chosen to be advertised must be 
- Alternate the use of network statements under router rip and IP-level interface configuration.
- Available Interfaces are: interface FastEthernet0/0, interface Ethernet1/0, interface Ethernet1/1, interface Ethernet1/2, interface Ethernet1/3, interface Serial2/0, interface Serial2/1, interface Serial2/2, interface Serial2/3, interface Serial2/4, interface Serial2/5, interface Serial2/6, interface Serial2/7 with the exception of subinterfaces like interface FastEthernet0/0.10


**Output only** the final, completed CLI configuration (no explanations), **Create 10 router configurations**
"""

# ─── 4) Call the ChatOpenAI model directly ────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=10000)
resp = llm.invoke([HumanMessage(content=prompt)])
full_cli = resp.text().strip()
parts = full_cli.split("```")
configs = []
for p in parts:
    cfg = p.strip("`\n ")
    if cfg:
        configs.append(cfg)

print(f"Found {len(configs)} config blocks")

out_dir = Path(r"Z:\Thesis\Generation Scripts&Results\RIP")
out_dir.mkdir(parents=True, exist_ok=True)

for i, cfg in enumerate(configs, start=41):
    # attempt to extract hostname, or fallback to index
    m = re.match(r"^hostname\s+(\S+)", cfg)
    name = m.group(1) if m else f"router{i}"
    path = out_dir / f"{name}_config.txt"
    path.write_text(cfg + "\n")
    print(f"Wrote config for {name} → {path}")