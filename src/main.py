from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import re
import os

def split_documents(journal_file: str):
    with open(journal_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by "## Month" pattern
    entries = re.split(r'\n(?=## \w+)', content)
    documents = [Document(text=entry.strip()) for entry in entries if entry.strip()]

    return documents


journal_path = "res/data/journals.md"
storage_dir = "res/storage"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

Settings.llm = Ollama(
    model="gemma3:4b",
    request_timeout=360.0,
    context_window=64_000,
)

if os.path.exists(storage_dir):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully!")
else:
    print("Creating new index...")
    docs = split_documents(journal_path)
    print(f"Processing {len(docs)} documents...")
    index = VectorStoreIndex.from_documents(docs)

    # Save the index
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Index saved to {storage_dir}")
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Retrieve 3 most relevant documents
    system_prompt="Du bist ein hilfreicher Assistent, der persönliche Tagebucheinträge analysiert. Antworte auf Deutsch und basiere deine Antworten nur auf den gegebenen Dokumenten."
)

questions = [
    "was sind meine pläne für die zukunft?",
    "was denke ich über AI?",
    "was sind meine lieblingsbücher?",
    "wie ist meine beziehung zu meiner mutter?"
]

for question in questions:

    response = query_engine.query(question)
    print(f"{question} -  Answer: {response}", end="\n\n")