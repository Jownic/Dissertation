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
    "Configure a login banner with the text \"Welcome to my C7200 UNAUTHORIZED ACCESS IS PROHIBITED. This device is monitored.\".",
    "Configure a failed-login banner with the text \"Nice try\".",
    "Set the login timeout to any value between 1 and 10000 seconds.",
    "Configure the RADIUS source interface to use an existing router interface.",
    "AAA accounting for connection events with the method list CONN-ACC.",
    "Configure AAA accounting for EXEC shell sessions with EXEC-ACC.",
    "Configure AAA accounting for network services with NET-ACC.",
    "Configure AAA accounting for system events with the default method list.",
    "Under `line con 0`, configure `login authentication LOGIN-LIST`.",
    "Under `line aux 0` configure `login authentication LOGIN-LIST",
    "Under `line vty 0 4` configure `login authentication EXEC-LIST",
    "Apply aaa authorization config-commands",
    "aaa authorization reverse-access default group tacacs+",
    "aaa authorization commands 15 default group tacacs+ if-authenticated",
    "aaa authentication login LOCAL-CASE local-case"
]

core_aaa = [
    "aaa new-model",
    "aaa authentication login LOGIN-LIST group tacacs+ local",
    "aaa authentication enable default group tacacs+ enable",
    "aaa accounting connection CONN-ACC start-stop group tacacs+",
    "aaa authorization exec EXEC-LIST group tacacs+",
    "aaa accounting exec EXEC-ACC start-stop group tacacs+",
    "aaa accounting network NET-ACC start-stop group tacacs+",
    "aaa accounting system default start-stop group tacacs+",
    "aaa accounting vrrs default start-stop group tacacs+",
    "aaa accounting delay-start",
    "Assign the TACACS+ server host IP to any address in the 10.0.0.0/24 subnet that has an interface associated with it.",
    "Configure an 11-character randomly generated alphanumeric secret for the TACACS+ server.",
    "service password-encryption",
    "Apply exec-timeout 10 0 to line vty 0 4",
    "Apply transport input ssh to line vty 0 4",
    "Apply access-class 10 in to line vty 0 4",
    "Apply access-list 10 permit with an ip range 10.0.0.0 0.0.0.255"

]


core_block = "\n".join(f"- {cmd}" for cmd in core_aaa)

# ─── 0) Boilerplate: load PDFs, chunk, embed, build FAISS (your existing code) ──
os.environ["OPENAI_API_KEY"] = "{INSERT OPEN API KEY HERE}"
client = OpenAI(api_key=os.getenv("{INSERT OPEN API KEY HERE}"))
folder = Path(r"Z:\Thesis\AConfig Guides\AAA")
raw_texts = []
for pdf in folder.rglob("sec-usr-aaa-15-s-book.pdf"):
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
docs = retriever.invoke("AAA")
ospf_docs = [
    d for d in docs 
    if d.metadata.get("source") == "sec-usr-aaa-15-s-book.pdf"
]
context_snippets = "\n\n".join(d.page_content for d in docs)

# ─── 2) Load your IOS template from file ──────────────────────────────────────
ios_template = Path(r"Z:\Thesis\ios_template.txt").read_text()

for i in range(1, 51):
    # 1) Sample 7 new extensions
    chosen = random.sample(extensions, k=9)
    ext_block = "\n".join(f"- {e}" for e in chosen)

# ─── 3) Build a single prompt with both context + template + instructions ─────
prompt = f"""
You are a network architect. 
**Use ONLY the following context** from the provided PDF manuals—do not rely on any other knowledge.
**Create 50 router configurations**
Output them one after the other
Each block MUST start with for example "hostname R1" "hostname R2" etc.
For each router, assign it a unique hostname (e.g. R1, R2…).  
For each router, assign it **unique** IP addresses for each interface
**Configure AAA**

=== **CORE AAA COMMANDS (Include on EVERY router):** ===
{core_block}
=== EXTENSIONS (Include all of the following on EVERY router ===
{ext_block}
=== CONTEXT SNIPPETS ===
{context_snippets}

=== IOS TEMPLATE ===
{ios_template}


=== INSTRUCTIONS ===
Fill in or extend the attached C7200 Router IOS template above so that it configures:
- Available Interfaces are: interface FastEthernet0/0, interface Ethernet1/0, interface Ethernet1/1, interface Ethernet1/2, interface Ethernet1/3, interface Serial2/0, interface Serial2/1, interface Serial2/2, interface Serial2/3, interface Serial2/4, interface Serial2/5, interface Serial2/6, interface Serial2/7 with the exception of subinterfaces like interface FastEthernet0/0.10
- DO NOT OMIT any command listed above—both CORE and OPTIONAL.
- If the context lacks any required command, leave that section blank and write UNKNOWN.
- List no shutdown on every interface 
- Assign each interface a unique /24 subnet in the 10.0.0.X/24 range
- **Output only** the final, completed CLI configuration (no explanations), **Create 10 router configurations**
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

out_dir = Path(r"Z:\Thesis\Generation Scripts&Results\AAA")
out_dir.mkdir(parents=True, exist_ok=True)

for i, cfg in enumerate(configs, start=1):
    # attempt to extract hostname, or fallback to index
    m = re.match(r"^hostname\s+(\S+)", cfg)
    name = m.group(1) if m else f"router{i}"
    path = out_dir / f"{name}_config.txt"
    path.write_text(cfg + "\n")
    print(f"Wrote config for {name} → {path}")