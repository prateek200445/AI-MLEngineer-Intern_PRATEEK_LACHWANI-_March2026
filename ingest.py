from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
from types import SimpleNamespace

from config import settings
from eligibility_engine import build_rule_profile, extract_course_codes, profile_to_dict


def _build_course_snippets(documents):
    snippets = {}
    for doc in documents:
        text = (getattr(doc, "page_content", "") or "")
        if not text.strip():
            continue

        lines = [ln.strip() for ln in text.splitlines()]
        for idx, line in enumerate(lines):
            line_codes = extract_course_codes(line)
            if not line_codes:
                continue

            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)
            snippet = " ".join(l for l in lines[start:end] if l)
            for code in line_codes:
                snippets.setdefault(code, []).append(snippet)

        # Also map prerequisite lines to nearby course code lines in the same page.
        for idx, line in enumerate(lines):
            if "prerequisite" not in line.lower():
                continue
            window_start = max(0, idx - 4)
            window_end = min(len(lines), idx + 2)
            window_text = " ".join(l for l in lines[window_start:window_end] if l)
            nearby_codes = extract_course_codes(window_text)
            for code in nearby_codes:
                snippets.setdefault(code, []).append(window_text)

    # Keep unique snippets per course in insertion order.
    for code, values in snippets.items():
        snippets[code] = list(dict.fromkeys(values))
    return snippets


def _build_prereq_chain_from_profiles(course_profiles):
    chain = {}
    for course_code, profile in course_profiles.items():
        prereq_candidates = (profile.get("any_of", []) or []) + (profile.get("all_of", []) or [])
        for pre in prereq_candidates:
            child = course_profiles.get(pre)
            if not child:
                continue
            child_prereqs = sorted(set((child.get("any_of", []) or []) + (child.get("all_of", []) or [])))
            if child_prereqs:
                chain.setdefault(course_code, {})[pre] = child_prereqs
    return chain


def generate_catalog_rules(documents, output_path):
    snippets_by_course = _build_course_snippets(documents)
    course_profiles = {}

    for code, snippets in snippets_by_course.items():
        pseudo_docs = [(SimpleNamespace(page_content=s), 1.0) for s in snippets]
        profile = build_rule_profile(
            target_course=code,
            docs=pseudo_docs,
            search_fn=None,
            depth=1,
        )
        course_profiles[code] = profile_to_dict(profile)

    chain = _build_prereq_chain_from_profiles(course_profiles)
    for code, mapping in chain.items():
        course_profiles[code]["prerequisite_chain"] = mapping

    payload = {
        "version": 1,
        "source_folder": settings.PDF_FOLDER,
        "course_count": len(course_profiles),
        "courses": course_profiles,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Catalog rules written: {output_path}")

pdf_folder = settings.PDF_FOLDER
# Load all PDFs
all_documents = []
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file_name)
        print(f"Loading: {file_name}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

print(f"Total PDFs loaded: {len(all_documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Larger chunks keep course title + prerequisite lines together.
    chunk_size=900,
    chunk_overlap=180,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)
texts = text_splitter.split_documents(all_documents)
print(f"Total text chunks created: {len(texts)}")

# Load the embedding model using settings
model_name = settings.EMBEDDING_MODEL
model_kwargs = {"device": settings.EMBEDDING_DEVICE}
encode_kwargs = {"normalize_embeddings": settings.EMBEDDING_NORMALIZE}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("  Embeddings loaded")

# Connect to Qdrant and push data
if settings.QDRANT_URL:
    url = settings.QDRANT_URL
else:
    url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

collection_name = settings.QDRANT_COLLECTION

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=settings.PREFER_GRPC,
    collection_name=collection_name
)

print(" Qdrant loaded and data ingested successfully")

rules_output = os.path.join(os.path.dirname(__file__), "catalog_rules.json")
generate_catalog_rules(all_documents, rules_output)
