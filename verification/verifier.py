# verification/verifier.py

from typing import List, Dict
from transformers import pipeline

# Initialize the entailment pipeline
entailment_pipeline = pipeline("text-classification", model="roberta-large-mnli")

def dndscore_verify(subclaim: str, decontext_claim: str, knowledge_source: List[Dict[str, str]]) -> bool:
    """
    Verifies whether the subclaim is supported by the search results using entailment.

    Args:
        subclaim: The original subclaim.
        decontext_claim: The decontextualized version of the subclaim.
        knowledge_source: A list of search result dictionaries containing 'title', 'snippet', and 'link'.

    Returns:
        True if the subclaim is supported, False otherwise.
    """
    try:
        for result in knowledge_source:
            premise = result.get('snippet', '')
            hypothesis = decontext_claim  # Use the decontextualized claim for verification

            # Perform entailment prediction
            prediction = entailment_pipeline(f"{premise} </s></s> {hypothesis}", truncation=True, max_length=512)
            label = prediction[0]['label']
            score = prediction[0]['score']

            if label == 'ENTAILMENT' and score > 0.7:  # Using entailment label and a threshold
                return True

        return False

    except Exception as e:
        print(f"Verification Error for subclaim '{subclaim}': {e}")
        return False