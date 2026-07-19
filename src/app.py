import json
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.train import predict as model_predict, FEATURE_COLUMNS

app = FastAPI(title="HormoneLab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This is not a diagnosis. Please consult a healthcare provider "
    "to confirm any hormonal health concerns."
)

DEFAULTS = {
    "age": 45,
    "fsh_level": 10.0,
    "lh_level": 10.0,
    "amh_level": 1.0,
    "estradiol_level": 50.0,
    "hormone_therapy_use": 0,
}

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
- fsh_level (number or null, FSH lab value in mIU/mL if explicitly mentioned)
- lh_level (number or null, LH lab value in mIU/mL if explicitly mentioned)
- amh_level (number or null, AMH lab value in ng/mL if explicitly mentioned)
- estradiol_level (number or null, estradiol lab value in pg/mL if explicitly mentioned)
- hormone_therapy_use ("yes", "no", or null)

Text: "{text}"
"""


def _extract_with_openai(text: str) -> dict:
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw).strip()
    return json.loads(raw)


def _extract_with_keywords(text: str) -> dict:
    t = text.lower()

    def find_number(pattern):
        m = re.search(pattern, t)
        return float(m.group(1)) if m else None

    age = None
    m = re.search(r"\b(\d{2})\s*(?:years old|yo|y/o|yrs)?\b", t)
    if m:
        val = int(m.group(1))
        if 12 <= val <= 100:
            age = val

    fsh = find_number(r"fsh[^\d]*(\d+(?:\.\d+)?)")
    lh = find_number(r"\blh[^\d]*(\d+(?:\.\d+)?)")
    amh = find_number(r"amh[^\d]*(\d+(?:\.\d+)?)")
    estradiol = find_number(r"estradiol[^\d]*(\d+(?:\.\d+)?)")

    hormone = None
    if "hormone therapy" in t or "hrt" in t or "estrogen" in t or "hormone-related" in t:
        hormone = "no" if ("never" in t or "not on" in t or "no hormone" in t) else "yes"

    return {
        "age": age,
        "fsh_level": fsh,
        "lh_level": lh,
        "amh_level": amh,
        "estradiol_level": estradiol,
        "hormone_therapy_use": hormone,
    }


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
        "expected_features": FEATURE_COLUMNS,
        "message": "POST /predict with a JSON body: {\"description\": \"...\"}",
    }


@app.post("/predict")
def predict_endpoint(req: SymptomRequest):
    text = req.description

    try:
        if _client:
            extracted = _extract_with_openai(text)
        else:
            extracted = _extract_with_keywords(text)
    except Exception:
        extracted = _extract_with_keywords(text)

    required = ["age", "fsh_level", "lh_level", "amh_level", "estradiol_level", "hormone_therapy_use"]
    missing = [f for f in required if extracted.get(f) is None]
    if len(missing) >= 5:
        return {
            "extracted": extracted,
            "prediction": None,
            "confidence": None,
            "explanation": (
                "Not enough information was provided to make a confident "
                f"prediction (missing: {', '.join(missing)}). " + DISCLAIMER
            ),
        }

    features = {
        "age": extracted.get("age") if extracted.get("age") is not None else DEFAULTS["age"],
        "fsh_level": extracted.get("fsh_level") if extracted.get("fsh_level") is not None else DEFAULTS["fsh_level"],
        "lh_level": extracted.get("lh_level") if extracted.get("lh_level") is not None else DEFAULTS["lh_level"],
        "amh_level": extracted.get("amh_level") if extracted.get("amh_level") is not None else DEFAULTS["amh_level"],
        "estradiol_level": extracted.get("estradiol_level") if extracted.get("estradiol_level") is not None else DEFAULTS["estradiol_level"],
        "hormone_therapy_use": 1 if extracted.get("hormone_therapy_use") == "yes" else 0,
    }

    label, confidence = model_predict(features)

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