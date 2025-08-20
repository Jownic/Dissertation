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
    "Under some of the EIGRP-enabled interface, configure: `ip hello-interval eigrp <AS> <TIME>`, where TIME is a random number from 1 to 60000. This must not be placed under `router eigrp`, only under interface stanzas.",
    "Enable log-neighbor-warnings with a seconds range randomly chosen from  1 to 65535",
    "Enable neighbor changes logging",
    "Under each EIGRP-enabled interface, configure: `ip hold-time eigrp <AS> <TIME>`, where TIME is a random number from 1 to 60000. This must not be placed under `router eigrp`, only under interface stanzas.",
    "Under the EIGRP autonomous system redistribute OSPF with an area of your choice (from 0 to 4294967295) i.e redistribute ospf 5",
    "Set a distance eigrp random number from 1 to 255 for internal and a random number from 1 to 255 for external. Do not use values above 255 — they are invalid and will break the configuration.",
    "On one randomly chosen EIGRP-enabled interface, under the interface stanza, add: ip bandwidth-percent eigrp with a random number from 10 to 100",
    "Set a passive interface",
    "Configure traffic share across-interfaces on the EIGRP AS",
    "Configure EIGRP variance with a random number in the range of 1 to 128",
    "Configure EIGRP maximum-paths to a random number in the range of 1 to 32",
    "Configure EIGRP stub randomly choosing between the options"
]

# ─── 0) Boilerplate: load PDFs, chunk, embed, build FAISS (your existing code) ──
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("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"))
folder = Path(r"Z:\Thesis\AConfig Guides\EIGRP")
raw_texts = []
for pdf in folder.rglob("ire-15-s-book.pdf"):
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
docs = retriever.invoke("EIGRP")
ospf_docs = [
    d for d in docs 
    if d.metadata.get("source") == "ire-15-s-book.pdf"
]
context_snippets = "\n\n".join(d.page_content for d in docs)

# ─── 2) Load your IOS template from file ──────────────────────────────────────
ios_template = Path(r"Z:\Thesis\ios_template.txt").read_text()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=10000)
config_counter = 1
for router_num in range(1, 51):
    # 1) Sample 7 new extensions
    chosen = random.sample(extensions, k=10)
    ext_block = "\n".join(f"{i+1}. {e}" for i, e in enumerate(chosen))

# ─── 3) Build a single prompt with both context + template + instructions ─────
    prompt = f"""
You are a senior network architect.
Use ONLY the following context from the provided PDF manuals—do not rely on any other knowledge.

Generate exactly five complete router configurations, one after another. For each router:
Hostname: R1...R5 (one per router)
Interfaces: Assign unique /24 subnets in the 192.168.X.0/24 range; configure IP addresses accordingly and issue no shutdown on each interface
EIGRP: choose a single random AS number (1-65535) and use it on all routers. Under router eigrp <AS> there should be 

router-id X.X.X.X ( X = router number).
metric weights (ex. metric weights 0 1 1 1 1 1)
distance eigrp <internal> <external>
1 - 4 neighbor <IP> <interface> statements per router; each neighbor IP and interface must share the same subnet and use the configured interface

Under each interface stanza, include any EIGRP-related interface commands (e.g ip hold-time eigrp) as required by the context. If a required command is missing from the context, write UNKNOWN in its place.

Available interface: interface FastEthernet0/0, interface Ethernet1/0, interface Ethernet1/1, interface Ethernet1/2, interface Ethernet1/3, interface Serial2/0, interface Serial2/1, interface Serial2/2, interface Serial2/3, interface Serial2/4, interface Serial2/5, interface Serial2/6, interface Serial2/7 with the exception of subinterfaces like interface FastEthernet0/0.10

=== EXTRA TASKS ===
{ext_block}

=== MANUAL ===
{context_snippets}

=== IOS TEMPLATE ===
{ios_template}

Output only the final configurations, with no explanations.
"""

# ─── 4) Call the ChatOpenAI model directly ────────────────────────────────────
    resp = llm.invoke([HumanMessage(content=prompt)])
    full_cli = resp.text().strip()
    parts = full_cli.split("```")
    configs = []
    for p in parts:
        cfg = p.strip("`\n ")
        if cfg:
            configs.append(cfg)

    print(f"Found {len(configs)} config blocks")

    out_dir = Path(r"Z:\Thesis\Generation Scripts&Results\EIGRP")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        path = out_dir / f"router{config_counter}_config.txt"
        path.write_text(cfg + "\n")
        print(f"Wrote config → {path}")
        config_counter += 1