import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from decomposition.decomposition_module import decompose_sentence
from decontextualization.decontext_module import decontextualize_with_llama3 as decontextualize
from core.core_module import CORE
from verification.verifier import dndscore_verify
from typing import List, Tuple, Dict, Union
import nltk
from templates.WebSearchAgent import search

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Placeholder for WebSearchAgent implementation
# from web_search_agent import WebSearchAgent  # You'll need to create this module

class FactCheckingPipeline:
    def __init__(self, entailment_model_name: str, similarity_model_name: str, bleached_claims_file: str, api_key: str = None, search_engine_id: str = None):
        """
        Initializes the FactCheckingPipeline with necessary components.
        
        Args:
            entailment_model_name: Name of the entailment model for CORE.
            similarity_model_name: Name of the similarity model for CORE.
            bleached_claims_file: Path to the bleached_claims.txt file.
            web_search_api_key: API key for the WebSearchAgent (optional if not using web search).
            search_engine_id: Search engine ID for the WebSearchAgent (optional if not using web search).
        """
        # Initialize CORE
        self.core = CORE(entailment_model_name, similarity_model_name)

        # Load bleached claims
        with open(bleached_claims_file, 'r', encoding='utf-8') as f:
            self.bleached_claims = [line.strip() for line in f if line.strip()]

        # Initialize Decontextualizer to use Llama 3 8B
        # self.decontextualizer = Decontextualizer(model_name="meta-llama/Llama-3-8b-chat-hf")
        # self.decontextualizer = Decontextualizer()

        # Initialize WebSearchAgent
        self.api_key = api_key

    def run_pipeline(self, text: str, threshold: float = 0.5) -> Tuple[float, List[Dict[str, Union[str, bool]]]]:
        """
        Runs the complete fact-checking pipeline.

        Args:
            text: The input text (LLM generation) to be fact-checked.
            threshold: The similarity threshold for CORE deduplication.

        Returns:
            A tuple containing:
                - The overall DnDScore (float).
                - A list of dictionaries, each representing a subclaim with its verification status.
        """

        # 1. Sentence Splitting
        sentences = self.split_into_sentences(text)
        print(f"Split into {len(sentences)} sentences.")

        # 2. Decomposition
        all_subclaims = []
        for sentence in sentences:
            subclaims = decompose_sentence(sentence)
            for fact in subclaims:
                # Reconstruct subclaim text from fact components
                subclaim_text = self.construct_subclaim_text(fact)
                all_subclaims.append((sentence, subclaim_text))
        print(f"Extracted {len(all_subclaims)} subclaims from decomposition.")

        # 3. Decontextualization
        decontextualized_subclaims = []
        for sentence, subclaim in all_subclaims:
            decontext_claim = decontextualize(subclaim, sentence, openai_key=self.api_key)
            decontextualized_subclaims.append(decontext_claim)
        print(f"Decontextualized {len(decontextualized_subclaims)} subclaims.")

        # 4. Deduplication (CORE)
        selected_indices = self.core.apply_core(decontextualized_subclaims, self.bleached_claims, threshold)
        filtered_subclaims = [decontextualized_subclaims[i] for i in selected_indices]
        print(f"Selected {len(filtered_subclaims)} unique and informative subclaims after CORE filtering.")

        # 5. Verification
        verified_facts = []
        for subclaim in filtered_subclaims:
            # Interact with WebSearchAgent to retrieve evidence
            search_results = json.loads(search(query=subclaim, num_results=5))
            
            # Verify the subclaim using the verification function
            is_supported = dndscore_verify(subclaim, subclaim, search_results)  # Pass the subclaim as both subclaim and decontext_claim
            
            verified_facts.append({
                "subclaim": subclaim,
                "is_supported": is_supported,
                "search_results": search_results  # Optionally include search results for transparency
            })
        print(f"Verified {len(verified_facts)} subclaims.")

        # 6. Calculate DnDScore
        num_supported_facts = sum(1 for fact in verified_facts if fact["is_supported"])
        dnd_score = num_supported_facts / len(verified_facts) if verified_facts else 0.0
        print(f"Calculated DnDScore: {dnd_score}")

        return dnd_score, verified_facts
    
    def construct_subclaim_text(self, fact: Dict[str, str]) -> str:
        """
        Constructs the subclaim text from the fact components.

        Args:
            fact: A dictionary containing 'subject', 'verb', 'direct_object', and 'prepositional_objects'.

        Returns:
            A string representing the subclaim.
        """
        subject = fact.get("subject", "")
        verb = fact.get("verb", "")
        direct_object = fact.get("direct_object", "")
        preps = ' '.join(fact.get("prepositional_objects", []))
        
        subclaim = f"{subject} {verb}"
        if direct_object:
            subclaim += f" {direct_object}"
        if preps:
            subclaim += f" {preps}"
        return subclaim.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits the given text into sentences using NLTK's sentence tokenizer.

        Args:
            text: The input text.

        Returns:
            A list of sentences.
        """
        return sent_tokenize(text)

    def formulate_search_query(self, subclaim: str, context: str) -> str:
        """
        Formulates a search query based on the subclaim and context.

        Args:
            subclaim: The subclaim to verify.
            context: The context of the subclaim.

        Returns:
            A search query string.
        """
        # Basic implementation: Combine subclaim and context as the query
        query = f"{subclaim} {context}"

        # Advanced implementation: Use an LLM to generate a more sophisticated query
        # prompt = f"Generate a search query to verify the following subclaim:\nSubclaim: {subclaim}\nContext: {context}\nQuery:"
        # query = generate_with_prompt_GPT(prompt, engine="gpt-4-0125-preview", max_tokens=50, openai_key=os.getenv("OPENAI_API_KEY"))

        return query

    def select_top_k_results(self, search_results: List[str], k: int) -> List[str]:
        """
        Selects the top k search results based on relevance.

        Args:
            search_results: A list of search result strings.
            k: The number of top results to select.

        Returns:
            A list of the top k search results.
        """
        # Basic implementation: Select the first k results
        return search_results[:k]

        # Advanced implementation: Use a ranking algorithm or model to select the most relevant results
        # ranked_results = rank_results(search_results, self.similarity_model)
        # return ranked_results[:k]

# Example usage (assuming you have loaded the entailment and similarity models)
# entailment_model = ... # Load your entailment model
# similarity_model = ... # Load your similarity model
# bleached_claims_file = "path/to/bleached_claims.txt"  # Replace with your bleached claims file
# pipeline = FactCheckingPipeline(entailment_model, similarity_model, bleached_claims_file)
# text = "Some long-form text generated by an LLM."
# dnd_score, verified_facts = pipeline.run_pipeline(text)
# print(f"DnDScore: {dnd_score}")
# for fact in verified_facts:
#     print(fact)