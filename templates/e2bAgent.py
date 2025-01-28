# e2bAgent.py

import requests
import time
import json
from typing import List, Dict, Any
import spacy
import logging
import yaml

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    filename='logs/e2bAgent.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class E2BAgent:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.api_url = config['coref']['api_url']
        self.hf_access_token = config['coref']['hf_access_token']
        self.headers = {"Authorization": f"Bearer {self.hf_access_token}"}
        self.max_retries = config['coref'].get('max_retries', 10)
        self.backoff_factor = config['coref'].get('backoff_factor', 5.0)
        
        # Initialize SpaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("SpaCy model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load SpaCy model: {e}")
            raise e

    def query_long_coref(self, paragraphs: List[str]) -> Dict[str, Any]:
        """
        Query the long-coref model via Hugging Face's Inference API.

        Args:
            paragraphs (List[str]): A list of paragraph strings.

        Returns:
            Dict[str, Any]: The JSON response from the model.
        """
        PARAGRAPH_DELIMITER = "\n\n"
        payload = {
            "inputs": PARAGRAPH_DELIMITER.join(paragraphs),
        }

        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                raise e

            if response.status_code == 503:
                # Model is loading
                error_message = response.json().get("error", "Service unavailable.")
                logging.info(f"Model loading: {error_message}. Retrying in {self.backoff_factor} seconds.")
                time.sleep(self.backoff_factor)
                retries += 1
                continue
            elif response.status_code == 200:
                logging.info("Coreference resolution successful.")
                return response.json()
            else:
                error_message = f"{response.status_code}: {response.json().get('error', 'Unknown error')}."
                logging.error(f"API Error: {error_message}")
                raise requests.HTTPError(error_message)
        
        logging.error("Max retries exceeded for coreference resolution.")
        raise requests.HTTPError("Max retries exceeded for coreference resolution.")

    def resolve_coreferences(self, coref_output: Dict[str, Any], paragraphs: List[str]) -> str:
        """
        Resolve coreferences based on the long-coref output.

        Args:
            coref_output (Dict[str, Any]): The JSON output from long-coref.
            paragraphs (List[str]): The original list of paragraphs.

        Returns:
            str: The text with coreferences resolved.
        """
        try:
            # Placeholder implementation
            # The actual implementation depends on the structure of coref_output
            # You need to adapt this based on the actual response structure
            resolved_paragraphs = []
            for para_idx, para in enumerate(paragraphs):
                resolved_paragraphs.append(para)  # Replace with actual resolution logic
            resolved_text = "\n\n".join(resolved_paragraphs)
            logging.info("Coreferences resolved in text.")
            return resolved_text
        except Exception as e:
            logging.error(f"Error in resolve_coreferences: {e}")
            return " ".join(paragraphs)  # Fallback to original text

    def extract_facts(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Extract atomic facts from the SpaCy Doc object.

        Args:
            doc (spacy.tokens.Doc): The SpaCy Doc object.

        Returns:
            List[Dict[str, Any]]: A list of extracted facts.
        """
        facts = []
        for sent in doc.sents:
            # Example extraction logic; replace with your actual method
            subject = ""
            verb = ""
            direct_object = ""
            prepositional_objects = []

            for token in sent:
                if token.dep_ == "nsubj":
                    subject = token.text
                if token.pos_ == "VERB":
                    verb = token.text
                if token.dep_ == "dobj":
                    direct_object = token.text
                if token.dep_ == "prep":
                    pobj = [child.text for child in token.children if child.dep_ == "pobj"]
                    if pobj:
                        prepositional_objects.append(f"{token.text.capitalize()} {' '.join(pobj)}")

            if subject and verb:
                fact = {
                    "subject": subject,
                    "verb": verb,
                    "direct_object": direct_object,
                    "prepositional_objects": prepositional_objects
                }
                facts.append(fact)
                logging.debug(f"Extracted fact: {fact}")
        
        logging.info(f"Total facts extracted: {len(facts)}")
        return facts

    def decompose_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Decompose a sentence into atomic facts using the long-coref model.

        Args:
            sentence (str): The sentence to decompose.

        Returns:
            List[Dict[str, Any]]: A list of atomic facts.
        """
        try:
            # Split sentence into paragraphs if necessary
            paragraphs = self.split_into_paragraphs(sentence)
            logging.info(f"Paragraphs split: {paragraphs}")
            
            # Query long-coref model
            coref_output = self.query_long_coref(paragraphs)
            
            # Process coref_output to resolve coreferences
            resolved_text = self.resolve_coreferences(coref_output, paragraphs)
            logging.info(f"Resolved Text: {resolved_text}")
            
            # Continue with existing decomposition logic using resolved_text
            doc = self.nlp(resolved_text)
            facts = self.extract_facts(doc)
            return facts
        except requests.HTTPError as e:
            logging.error(f"Coreference resolution failed with HTTPError: {e}. Proceeding without coref.")
        except Exception as e:
            logging.error(f"Coreference resolution failed with Exception: {e}. Proceeding without coref.")
        
        # Fallback: Proceed without coreference resolution
        doc = self.nlp(sentence)
        facts = self.extract_facts(doc)
        return facts

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split the text into paragraphs. Customize if needed.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of paragraph strings.
        """
        # Simple split by double newline; adjust as necessary
        paragraphs = text.strip().split("\n\n")
        logging.debug(f"Text split into paragraphs: {paragraphs}")
        return paragraphs

if __name__ == "__main__":
    agent = E2BAgent(config_path='config/config.yaml')
    test_sentence = "He gave the book to the girl in the library."
    extracted_facts = agent.decompose_sentence(test_sentence)
    print(json.dumps(extracted_facts, indent=2))
