# decontextualization/decontext_module.py

import os
import requests
import json
import spacy

nlp = spacy.load("en_core_web_sm")

API_URL = os.environ.get("GROQ_API_ENDPOINT")
API_KEY = os.environ.get("GROQ_API_KEY")

def decontextualize_with_llama3(subclaim: str, context: str) -> str:
    """
    Decontextualizes a subclaim using the Llama 3 8B model via the Groq API.

    Args:
        subclaim: The subclaim to decontextualize.
        context: The original sentence containing the subclaim.

    Returns:
        The decontextualized subclaim.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Decontextualize the subclaim to make it standalone, ensuring grammatical correctness and maintaining the original meaning."},
            {"role": "user", "content": f"Context: {context}\nSubclaim: {subclaim}\n\nDecontextualize the subclaim."}
        ],
        "max_tokens": 100,
        "temperature": 0.3,
        "n": 1,
        "stop": ["\n"]
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        decontextualized_claim = response.json()['choices'][0]['message']['content'].strip()
        return decontextualized_claim
    except Exception as e:
        print(f"Error during decontextualization with Llama 3: {e}")
        return subclaim  # Fallback: return the original subclaim

def fallback_decontextualize(subclaim: str, context: str) -> str:
    """
    Fallback method for decontextualization using SpaCy's pronoun resolution.

    Args:
        subclaim: The subclaim to decontextualize.
        context: The original sentence containing the subclaim.

    Returns:
        The decontextualized subclaim.
    """
    doc = nlp(context)
    
    # Basic pronoun resolution
    if subclaim.lower().startswith("he "):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                subclaim = subclaim.replace("He", ent.text, 1)
                break
    elif subclaim.lower().startswith("she "):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                subclaim = subclaim.replace("She", ent.text, 1)
                break
    elif subclaim.lower().startswith("it "):
        for ent in doc.ents:
            if ent.label_ not in ["PERSON", "NORP", "GPE"]:
                subclaim = subclaim.replace("It", ent.text, 1)
                break
    elif subclaim.lower().startswith("they "):
        # Example handling for 'they'
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "NORP"]:
                subclaim = subclaim.replace("They", ent.text, 1)
                break
    
    return subclaim