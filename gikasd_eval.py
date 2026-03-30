import os
import json
import requests
import pandas as pd

# Make sure Giskard is installed
try:
    import giskard
    from giskard import Model, Dataset
except ImportError:
    raise ImportError("Install Giskard first: pip install giskard")

# ===================== CONFIG =====================
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
DATASET_PATH = r"D:\qdrant_ping\collections\ragas_eval_data.json"

if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable.")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

PROMPT_TEMPLATE = "{input}\nAnswer:"  # Optional template for formatting input

# ===================== GEMINI CLIENT =====================
def generate(prompt: str, temperature: float = 0.0) -> str:
    """Generate using local Ollama CLI (expects `ollama` installed and running).

    Falls back to returning the raw prompt result on error.
    """
    import subprocess
    model = MODEL_NAME or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    cli = os.environ.get("OLLAMA_CLI", "ollama")
    try:
        proc = subprocess.run(
            [cli, "run", model],
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(os.environ.get("OLLAMA_TIMEOUT") or 300),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "Ollama CLI error")
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        # Retry once with extended timeout
        try:
            proc = subprocess.run(
                [cli, "run", model],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=int((os.environ.get("OLLAMA_TIMEOUT") or 300) * 2),
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "Ollama CLI error on retry")
            return proc.stdout.strip()
        except Exception:
            return f"<ERROR: ollama timeout on retry for prompt: {prompt[:80]}...>"
    except Exception:
        # Last-resort: return a debug string so evaluation can continue
        return f"<ERROR: failed to generate via ollama for prompt: {prompt[:80]}...>"

# ===================== LOAD AND PROCESS DATASET =====================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Handle cases where JSON is a dict containing a list
if isinstance(raw_data, dict):
    # Try to auto-detect the first list in the dict
    for v in raw_data.values():
        if isinstance(v, list):
            raw_dataset = v
            break
    else:
        raise ValueError("No list found in JSON file for dataset")
elif isinstance(raw_data, list):
    raw_dataset = raw_data
else:
    raise ValueError("JSON file must be a list or dict containing a list")

# Build DataFrame
rows = []

# Accept multiple possible key names for inputs and expected outputs
INPUT_KEYS = ["input", "prompt", "question", "text", "query", "source"]
EXPECTED_KEYS = ["expected", "answer", "label", "target", "output"]

for i, row in enumerate(raw_dataset):
    inp = None
    expected = None

    if isinstance(row, dict):
        # try canonical keys first, then fallback to alternatives
        for k in INPUT_KEYS:
            if k in row and row[k]:
                inp = row[k]
                break
        for k in EXPECTED_KEYS:
            if k in row and row[k] is not None:
                expected = row[k]
                break

        # Some datasets nest content under 'data' or 'example'
        if inp is None:
            for alt in ("data", "example", "item"):
                if alt in row and isinstance(row[alt], dict):
                    for k in INPUT_KEYS:
                        if k in row[alt] and row[alt][k]:
                            inp = row[alt][k]
                            break
                    for k in EXPECTED_KEYS:
                        if k in row[alt] and row[alt][k] is not None:
                            expected = row[alt][k]
                            break
                    if inp or expected:
                        break

    elif isinstance(row, (list, tuple)) and len(row) >= 2:
        inp, expected = row[0], row[1]
    elif isinstance(row, str):
        inp = row
        expected = None

    # normalize simple dict values that may be lists (take first element)
    if isinstance(inp, list):
        inp = inp[0] if inp else None
    if isinstance(expected, list):
        expected = expected[0] if expected else None

    if inp and expected:
        prompt = PROMPT_TEMPLATE.format(input=inp) if PROMPT_TEMPLATE else inp
        rows.append({"input": prompt, "expected": expected})
    else:
        # keep a very small sample in debug if needed later
        continue

if not rows:
    # helpful debug info to diagnose input JSON format
    sample_preview = raw_dataset[:10] if isinstance(raw_dataset, list) else list(raw_dataset.items())[:10]
    print("DEBUG: No valid rows parsed from dataset. Sample of raw data (first 10 items):")
    print(json.dumps(sample_preview, indent=2, ensure_ascii=False))
    raise ValueError(
        "No valid rows found in dataset. Check JSON format and keys. "
        "Supported input keys: " + ", ".join(INPUT_KEYS) + ". "
        "Supported expected keys: " + ", ".join(EXPECTED_KEYS) + "."
    )

df = pd.DataFrame(rows)
print(f"Dataset loaded with {len(df)} samples.")
print(df.head())

# ===================== CREATE GISKARD DATASET =====================
def generate(prompt: str, temperature: float = 0.0) -> str:
    """Generate using Gemini / Google Generative API over HTTP.

    Reads `GOOGLE_API_KEY` (or `API_KEY`) and `API_URL` (or `GOOGLE_API_URL`) from env.
    Returns a debug string on failure so evaluation can continue.
    """
    import os
    try:
        import requests

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
        api_url = os.environ.get("API_URL") or os.environ.get("GOOGLE_API_URL")
        if not api_key or not api_url:
            return "<ERROR: GOOGLE_API_KEY or API_URL not set>"

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "temperature": temperature}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # parse common response shapes
        if isinstance(data, dict):
            if "output" in data and data["output"]:
                return data["output"]
            if "text" in data and data["text"]:
                return data["text"]
            if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                first = data["candidates"][0]
                if isinstance(first, dict):
                    return first.get("output") or first.get("text") or str(first)
                return str(first)

        return str(data)
    except Exception as e:
        return f"<ERROR: failed to generate via gemini: {e}>"
    print(f"Expected: {r['expected']}")
