# decontextualization/decontext_module.py

import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

API_URL = os.environ.get("GROQ_API_ENDPOINT")
API_KEY = os.environ.get("GROQ_API_KEY")

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
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

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Decontextualize the subclaim to make it standalone, ensuring grammatical correctness and maintaining the original meaning."
        },
        {
            "role": "user",
            "content": f"Context: {context}\nSubclaim: {subclaim}\n\nDecontextualize the subclaim."
        }
    ]

    data = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.3,
        "n": 1,
        "stop": ["\n"]
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            decontextualized_claim = response_json['choices'][0]['message']['content'].strip()
            return decontextualized_claim
        else:
            print(f"Unexpected response format: {response_json}")
            return subclaim

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return subclaim  # Fallback: return the original subclaim
    except (KeyError, IndexError) as e:
        print(f"Error in parsing response: {e}")
        return subclaim  # Fallback: return the original subclaim
    except Exception as e:
        print(f"Unexpected error during decontextualization with Llama 3: {e}")
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