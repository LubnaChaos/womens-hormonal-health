import json
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

from src.train import predict as model_predict

app = FastAPI(title="Women's Hormonal Health — Menopause Status Predictor")

DISCLAIMER = (
    "This is not a diagnosis. Please consult a healthcare provider "
    "to confirm any hormonal health concerns."
)

# --- OpenAI client is optional -------------------------------------------
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_client = None
if _OPENAI_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=_OPENAI_KEY)
    except Exception:
        _client = None


class SymptomRequest(BaseModel):
    description: str


EXTRACTION_PROMPT = """Extract these fields from the text as JSON only, no other text, no markdown.
Use null for anything not mentioned:
- age (integer or null)
- fsh_level (number or null, only if an FSH lab value is explicitly mentioned)
- hormone_therapy_use ("yes", "no", or null)

Text: "{text}"
"""


def _extract_with_openai(text: str) -> dict:
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
    )
    raw = resp.choices[0].message.content.strip()
    # strip accidental markdown fences if present
    raw = re.sub(r"^```(json)?|```$", "", raw).strip()
    return json.loads(raw)


def _extract_with_keywords(text: str) -> dict:
    """Fallback extractor: no OpenAI needed. Simple, transparent rules."""
    t = text.lower()
    age = None
    m = re.search(r"\b(\d{2})\s*(?:years old|yo|y/o|yrs)?\b", t)
    if m:
        val = int(m.group(1))
        if 12 <= val <= 100:
            age = val
    fsh = None
    m = re.search(r"fsh[^\d]*(\d+(?:\.\d+)?)", t)
    if m:
        fsh = float(m.group(1))
    hormone = None
    if "hormone therapy" in t or "hrt" in t or "estrogen" in t:
        hormone = "no" if ("never" in t or "not on" in t or "no hormone" in t) else "yes"
    return {"age": age, "fsh_level": fsh, "hormone_therapy_use": hormone}


def _explain_with_openai(label, confidence) -> str:
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"In one or two friendly sentences, explain this result to a "
                f"non-technical person: prediction={label}, confidence={confidence:.0%}. "
                f"Do not add medical advice beyond suggesting a healthcare provider."
            ),
        }],
    )
    return resp.choices[0].message.content.strip()


def _explain_templated(label, confidence) -> str:
    return (
        f"Based on the information provided, the pattern is most consistent "
        f"with being {label} (confidence {confidence:.0%})."
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "openai_enabled": _client is not None,
        "message": "POST /predict with a JSON body: {\"description\": \"...\"}",
    }


@app.post("/predict")
def predict_endpoint(req: SymptomRequest):
    text = req.description

    # Step 1: free text -> structured features (OpenAI if available, else keywords)
    try:
        if _client:
            extracted = _extract_with_openai(text)
        else:
            extracted = _extract_with_keywords(text)
    except Exception:
        # if OpenAI returns something unparseable, fall back gracefully
        extracted = _extract_with_keywords(text)

    # Step 2: no-call case — too little information for a confident prediction
    required = ["age", "fsh_level", "hormone_therapy_use"]
    missing = [f for f in required if extracted.get(f) is None]
    if len(missing) >= 3:
        return {
            "extracted": extracted,
            "prediction": None,
            "confidence": None,
            "explanation": (
                "Not enough information was provided to make a confident "
                f"prediction (missing: {', '.join(missing)}). " + DISCLAIMER
            ),
        }

    # Step 3: fill remaining gaps with documented neutral defaults
    features = {
        "age": extracted.get("age") or 45,
        "fsh_level": extracted.get("fsh_level") if extracted.get("fsh_level") is not None else 10.0,
        "hormone_therapy_use": 1 if extracted.get("hormone_therapy_use") == "yes" else 0,
    }

    # Step 4: run the trained model (local — never OpenAI)
    label, confidence = model_predict(features)

    # Step 5: plain-language explanation
    if _client:
        try:
            explanation = _explain_with_openai(label, confidence)
        except Exception:
            explanation = _explain_templated(label, confidence)
    else:
        explanation = _explain_templated(label, confidence)

    return {
        "extracted": extracted,
        "features_used": features,
        "prediction": label,
        "confidence": confidence,
        "explanation": explanation + " " + DISCLAIMER,
    }
