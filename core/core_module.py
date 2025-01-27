# core/core_module.py
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class CORE:
    def __init__(self, entailment_model_name: str = 'roberta-large-mnli', similarity_model: str = 'all-mpnet-base-v2'):
        # Load the entailment model and tokenizer
        self.entailment_tokenizer = AutoTokenizer.from_pretrained(entailment_model_name)
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(entailment_model_name)
        self.entailment_model.eval()  # Set to evaluation mode

        # Load the similarity model
        self.similarity_model = SentenceTransformer(similarity_model)

    def calculate_informativeness_weights(self, subclaims: List[str], bleached_claims: List[str]) -> List[float]:
        """
        Calculates informativeness weights for each subclaim based on Conditional Pairwise Mutual Information (CPMI).

        Args:
            subclaims: A list of subclaims.
            bleached_claims: A list of bleached claims for the domain.

        Returns:
            A list of informativeness weights corresponding to the subclaims.
        """
        weights = []
        for subclaim in subclaims:
            min_entailment = 1.0
            for bleached_claim in bleached_claims:
                entail_prob = self.get_entailment_probability(bleached_claim, subclaim)
                min_entailment = min(min_entailment, entail_prob)
            
            informativeness = -np.log(min_entailment) if min_entailment > 0 else 0.0  # Handle log(0)
            weights.append(informativeness)
        return weights

    def get_entailment_probability(self, premise: str, hypothesis: str) -> float:
        """
        Calculates the probability that the premise entails the hypothesis.

        Args:
            premise: The premise string.
            hypothesis: The hypothesis string.

        Returns:
            The probability of entailment (between 0 and 1).
        """
        # Tokenize input
        inputs = self.entailment_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.entailment_model(**inputs)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

        # MNLI labels: 0 - contradiction, 1 - neutral, 2 - entailment
        entailment_prob = probabilities[2]  # Probability of entailment
        return float(entailment_prob)

    def calculate_similarity(self, claim1: str, claim2: str) -> float:
        """
        Calculates the cosine similarity between two claims.

        Args:
            claim1: The first claim string.
            claim2: The second claim string.

        Returns:
            The cosine similarity score (between 0 and 1).
        """
        embeddings = self.similarity_model.encode([claim1, claim2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity
    
    def select_core_subclaims(self, subclaims: List[str], informativeness_weights: List[float], threshold: float = 0.5) -> List[int]:
        """
        Selects a subset of subclaims that are unique and informative using a greedy approach.

        Args:
            subclaims: A list of subclaims.
            informativeness_weights: A list of informativeness weights corresponding to the subclaims.
            threshold: The similarity threshold for considering subclaims as duplicates.

        Returns:
            A list of indices of selected subclaims.
        """
        # Pair subclaims with their weights and original indices
        subclaims_with_weights_and_indices = list(enumerate(zip(subclaims, informativeness_weights)))
        # Sort subclaims by informativeness in descending order
        subclaims_sorted = sorted(subclaims_with_weights_and_indices, key=lambda x: x[1][1], reverse=True)
        
        selected_indices = []
        selected_embeddings = []

        for idx, (subclaim, weight) in subclaims_sorted:
            is_unique = True
            embedding = self.similarity_model.encode(subclaim)
            
            for selected_embedding in selected_embeddings:
                similarity = np.dot(embedding, selected_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(selected_embedding))
                if similarity > threshold:
                    is_unique = False
                    break

            if is_unique:
                selected_embeddings.append(embedding)
                selected_indices.append(idx)
        
        return selected_indices

    def apply_core(self, subclaims: List[str], bleached_claims: List[str], threshold: float = 0.5) -> List[int]:
        """
        Applies the CORE algorithm to filter subclaims.

        Args:
            subclaims: A list of subclaims.
            bleached_claims: A list of bleached claims for the domain.
            threshold: The similarity threshold for deduplication.

        Returns:
            A list of indices of filtered subclaims.
        """
        weights = self.calculate_informativeness_weights(subclaims, bleached_claims)
        selected_indices = self.select_core_subclaims(subclaims, weights, threshold)
        return selected_indices