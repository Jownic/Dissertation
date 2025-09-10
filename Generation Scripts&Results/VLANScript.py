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

# ─── 0) Boilerplate: load PDFs, chunk, embed, build FAISS (your existing code) ──
os.environ["OPENAI_API_KEY"] = "{INSERT OPEN API KEY HERE}"
client = OpenAI(api_key=os.getenv("{INSERT OPEN API KEY HERE}"))
folder = Path(r"Z:\Thesis\AConfig Guides\VLANs")
raw_texts = []
for pdf in folder.rglob("lsw-15-s-book.pdf"):
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
docs = retriever.invoke("VLAN")
vlan_docs = [
    d for d in docs 
    if d.metadata.get("source") == "lsw-15-s-book.pdf"
]
context_snippets = "\n\n".join(d.page_content for d in docs)

# ─── 2) Load your IOS template from file ──────────────────────────────────────
ios_template = Path(r"Z:\Thesis\ios_template.txt").read_text()

# ─── 3) Build a single prompt with both context + template + instructions ─────
prompt = f"""
You are a network architect.  
**Use ONLY the following context** from the provided PDF manuals—do not rely on any other knowledge.
Output them one after the other
Choose a random number from 1 to 6 then create that amount of VLANs on the chosen router
Choose a random number from 0 to 4095 for the VLAN ID
Each block MUST start with for example "hostname R1" "hostname R2" etc.
For each router, assign it a unique hostname (e.g. R1, R2…).  
Use only subnets in the 192.168.*.*/24 space for VLANs, Make sure to choose a random number from 1-254 to replace the "*" in the ip's
For each router, assign it **unique** IP addresses for each link and VLAN interface
**Use ONLY the following context** from the provided PDF manuals—do not rely on any other knowledge.
If the context lacks any required command, leave that section blank and write UNKNOWN.

=== CONTEXT SNIPPETS ===
{context_snippets}

=== IOS TEMPLATE ===
{ios_template}

=== INSTRUCTIONS ===
Fill in or extend the attached C7200 Router IOS template above so that it configures:
- VLANS
- Layer-3 inter-VLAN routing
- Available Interfaces are: interface FastEthernet0/0, interface Ethernet1/0, interface Ethernet1/1, interface Ethernet1/2, interface Ethernet1/3, interface Serial2/0, interface Serial2/1, interface Serial2/2, interface Serial2/3, interface Serial2/4, interface Serial2/5, interface Serial2/6, interface Serial2/7 with the exception of subinterfaces like interface FastEthernet0/0.10

**Output only** the final, completed CLI configuration (no explanations), All VLANS must be put into subinterfaces and **ONLY the interfaces mentioned are to be used Create 5 router configurations**. 
"""

# ─── 4) Call the ChatOpenAI model directly ────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=10000)
resp = llm.invoke([HumanMessage(content=prompt)])
print(resp.text())
full_cli = resp.text().strip()
parts = full_cli.split("```")
configs = []
for p in parts:
    cfg = p.strip("`\n ")
    if cfg:
        configs.append(cfg)

print(f"Found {len(configs)} config blocks")

out_dir = Path(r"Z:\Thesis\Generation Scripts&Results\VLANS")
out_dir.mkdir(parents=True, exist_ok=True)

for i, cfg in enumerate(configs, start=81):
    # attempt to extract hostname, or fallback to index
    m = re.match(r"^hostname\s+(\S+)", cfg)
    name = m.group(1) if m else f"router{i}"
    path = out_dir / f"{name}_config.txt"
    path.write_text(cfg + "\n")
    print(f"Wrote config for {name} → {path}")