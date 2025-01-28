import spacy
from spacy.tokens import Token
import requests
import json
import os

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

def long_coref_resolution(text):
    """
    Perform coreference resolution on long documents using the long-coref model from Hugging Face.

    Args:
        text (str): The input text (long document).

    Returns:
        str: The text with coreferences resolved.
    """
    API_URL = "https://api-inference.huggingface.co/models/kwang2049/long-coref"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"}

    def query(payload):
        data = json.dumps(payload)
        response = requests.post(API_URL, headers=headers, data=data, timeout=300)
        return response.json()

    # Split the text into smaller chunks for processing
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Ensure not to exceed model's maximum token limit
    max_length = 512  # Adjust based on the model's limit
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(nlp(sentence))  # Count tokens in the sentence
        if current_length + sentence_length < max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk

    resolved_text = ""
    for chunk in chunks:
        output = query({"inputs": chunk})
        
        # Check for errors in the response
        if "error" in output:
            print(f"Error processing chunk: {output['error']}")
            resolved_text += chunk  # Append the original chunk if error occurs
            continue
        
        # Extract resolved text from the output
        resolved_text += output.get("resolved_text", chunk)

    return resolved_text

def get_full_object(token):
    """
    Recursively get the full object phrase, including prepositional phrases and modifiers.

    Args:
        token (spacy.Token): The object token.

    Returns:
        str: The full object phrase.
    """
    obj_phrase = token.text
    # Include compound nouns, adjectival modifiers, determiners, and appositives
    modifiers = [child.text for child in token.lefts if child.dep_ in ('compound', 'amod', 'det', 'appos')]
    if modifiers:
        obj_phrase = ' '.join(modifiers) + ' ' + obj_phrase

    # Include prepositional phrases
    preps = [child for child in token.children if child.dep_ == 'prep']
    for prep in preps:
        prep_phrase = prep.text
        for pobj in prep.children:
            if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON'):
                prep_obj = get_full_object(pobj)
                prep_phrase += f" {prep_obj}"
                obj_phrase += f" {prep_phrase}"
    return obj_phrase

def remove_trailing_punctuation(s):
    """
    Remove trailing punctuation from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The string with trailing punctuation removed.
    """
    if s and s[-1] in [',', ';']:
        return s[:-1].strip()
    return s

def decompose_sentence(sentence: str) -> List[str]:
    """
    Decompose a sentence into atomic facts by extracting comprehensive subject-verb-object relationships,
    including various object types and handling complex sentence structures.

    Args:
        sentence (str): The sentence to decompose.

    Returns:
        list: A numbered list of atomic facts extracted from the sentence.
              Returns an empty list if decomposition isn't possible.
    """
    try:
        # Resolve coreferences using long-coref model
        resolved_text = long_coref_resolution(sentence)
        resolved_doc = nlp(resolved_text)
    except Exception as e:
        print(f"Error resolving coreferences in sentence: {e}")
        return []

    facts = []
    current_subject = None

    for sent in resolved_doc.sents:
        has_root = any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in sent)
        if not has_root:
            print(f"Skipping sentence with no main verb: {sent.text}")
            continue

        for token in sent:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"):
                verb_main = token.text
                verb_aux = ' '.join([child.text for child in token.children if child.dep_ in ('aux', 'auxpass')])
                full_verb = f"{verb_aux} {verb_main}".strip() if verb_aux else verb_main

                subject = next((child.text for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')), None)
                if subject:
                    subject_modifiers = [w.text for w in token.children if w.dep_ in ('det', 'amod', 'compound', 'appos')]
                    subject = ' '.join(subject_modifiers + [subject])
                else:
                    subject = current_subject

                if subject:
                    current_subject = subject

                direct_objects = [get_full_object(child) for child in token.children if child.dep_ in ('dobj', 'attr', 'oprd', 'iobj')]
                prepositional_objects = [f"{child.text.capitalize()} {get_full_object(pobj)}"
                                         for child in token.children if child.dep_ == 'prep'
                                         for pobj in child.children if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON')]

                # Handle relative clauses attached to the main verb's object
                for child in token.children:
                    if child.dep_ == 'relcl':
                        relative_fact = decompose_relative_clause(child)
                        if relative_fact:
                            facts.append(relative_fact)

                # Add main verb fact
                if subject and full_verb:
                    fact = {
                        "subject": subject.strip(','),
                        "verb": full_verb.lower(),
                        "direct_object": ', '.join(direct_objects).capitalize() if direct_objects else "",
                        "prepositional_objects": prepositional_objects if prepositional_objects else []
                    }
                    if fact["subject"] and fact["verb"]:
                        facts.append(fact)

                # Handle conjunct verbs sharing the same subject
                for conj in token.conjuncts:
                    if conj.pos_ != "VERB":
                        continue
                    conj_verb = conj.text
                    conj_aux = ' '.join([child.text for child in conj.children if child.dep_ in ('aux', 'auxpass')])
                    full_conj_verb = f"{conj_aux} {conj_verb}".strip() if conj_aux else conj_verb

                    conj_direct_objects = [get_full_object(child) for child in conj.children if child.dep_ in ('dobj', 'attr', 'oprd', 'iobj')]
                    conj_prepositional_objects = [f"{child.text.capitalize()} {get_full_object(pobj)}" for child in conj.children if child.dep_ == 'prep' for pobj in child.children if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON')]

                    if subject and full_conj_verb:
                        conj_fact = {
                            "subject": subject.strip(','),
                            "verb": full_conj_verb.lower(),
                            "direct_object": ', '.join(conj_direct_objects).capitalize() if conj_direct_objects else "",
                            "prepositional_objects": conj_prepositional_objects if conj_prepositional_objects else []
                        }
                        if conj_fact["subject"] and conj_fact["verb"]:
                            facts.append(conj_fact)

    # Post-processing for enhanced atomicity
    atomic_facts = []
    for fact in facts:
        atomic_facts.extend(split_complex_fact(fact, resolved_doc))

    # Convert facts to a numbered list format
    numbered_facts = [f"{i+1}. {format_fact(fact)}" for i, fact in enumerate(atomic_facts) if fact["subject"] and fact["verb"]]

    return numbered_facts

def decompose_relative_clause(token):
    """
    Extract facts from a relative clause.

    Args:
        token (spacy.Token): The token representing the relative clause.

    Returns:
        dict: A dictionary representing the extracted fact, or None if no fact is found.
    """
    relative_verb = token.text
    relative_subject = None
    for rel_child in token.children:
        if rel_child.dep_ in ('nsubj', 'nsubjpass'):
            relative_subject = rel_child.text
            rel_modifiers = [w.text for w in rel_child.lefts if w.dep_ in ('det', 'amod', 'compound', 'appos')]
            if rel_modifiers:
                relative_subject = ' '.join(rel_modifiers + [relative_subject])
            break

    if relative_subject:
        relative_direct_objects = [get_full_object(rel_child) for rel_child in token.children if rel_child.dep_ in ('dobj', 'attr', 'oprd', 'iobj')]
        relative_prepositional_objects = [f"{rel_child.text.capitalize()} {get_full_object(rel_pobj)}" for rel_child in token.children if rel_child.dep_ == 'prep' for rel_pobj in rel_child.children if rel_pobj.dep_ == 'pobj' and rel_pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON')]

        relative_fact = {
            "subject": relative_subject.strip(','),
            "verb": relative_verb.lower(),
            "direct_object": ', '.join(relative_direct_objects).capitalize() if relative_direct_objects else "",
            "prepositional_objects": relative_prepositional_objects if relative_prepositional_objects else []
        }
        if relative_fact["subject"] and relative_fact["verb"]:
            return relative_fact
    return None

def split_complex_fact(fact, doc):
    """
    Splits a complex fact into multiple atomic facts based on conjunctions and nested clauses.

    Args:
        fact (dict): A fact dictionary with 'subject', 'verb', 'direct_object', and 'prepositional_objects'.
        doc (spacy.Doc): The SpaCy document for the original sentence.

    Returns:
        list: A list of atomic fact dictionaries.
    """
    atomic_facts = []
    
    # Check for conjunctions in the direct object
    if ' and ' in fact.get('direct_object', ''):
        parts = fact['direct_object'].split(' and ')
        for part in parts:
            new_fact = fact.copy()
            new_fact['direct_object'] = part.strip()
            atomic_facts.extend(split_complex_fact(new_fact, doc))  # Recursive call to handle further splitting
        return atomic_facts

    # Check for conjunctions in prepositional objects
    prepositional_objects = fact.get('prepositional_objects', [])
    if prepositional_objects:
        new_prepositional_objects = []
        for prep_obj in prepositional_objects:
            if ' and ' in prep_obj:
                parts = prep_obj.split(' and ')
                for part in parts:
                    new_prepositional_objects.append(part.strip())
            else:
                new_prepositional_objects.append(prep_obj)
        if new_prepositional_objects != prepositional_objects:
            new_fact = fact.copy()
            new_fact['prepositional_objects'] = new_prepositional_objects
            atomic_facts.extend(split_complex_fact(new_fact, doc))  # Recursive call to handle further splitting
            return atomic_facts

    # Check for relative clauses in direct object
    if ' that ' in fact.get('direct_object', '') or ' which ' in fact.get('direct_object', ''):
        parts = fact['direct_object'].split(' that ')
        main_object = parts[0].strip()
        if len(parts) > 1:
            relative_clause = 'that ' + ' that '.join(parts[1:])
            relative_doc = nlp(relative_clause)
            for token in relative_doc:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    relative_subject = None
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'nsubjpass'):
                            relative_subject = child.text
                            rel_modifiers = [w.text for w in child.lefts if w.dep_ in ('det', 'amod', 'compound', 'appos')]
                            if rel_modifiers:
                                relative_subject = ' '.join(rel_modifiers + [relative_subject])
                            break
                    relative_direct_objects = []
                    relative_prepositional_objects = []
                    for rel_child in token.children:
                        if rel_child.dep_ in ('dobj', 'attr', 'oprd', 'iobj'):
                            obj = get_full_object(rel_child)
                            relative_direct_objects.append(obj)
                        elif rel_child.dep_ == 'prep':
                            for rel_pobj in rel_child.children:
                                if rel_pobj.dep_ == 'pobj' and rel_pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON'):
                                    prep_obj = get_full_object(rel_pobj)
                                    relative_prepositional_objects.append(f"{rel_child.text.capitalize()} {prep_obj}")
                    if relative_subject:
                        relative_fact = {
                            "subject": relative_subject.strip(','),
                            "verb": token.text.lower(),
                            "direct_object": ', '.join(relative_direct_objects).capitalize() if relative_direct_objects else "",
                            "prepositional_objects": relative_prepositional_objects if relative_prepositional_objects else []
                        }
                        atomic_facts.append(relative_fact)
        main_fact = fact.copy()
        main_fact['direct_object'] = main_object
        atomic_facts.append(main_fact)
    else:
        atomic_facts.append(fact)

    return atomic_facts

def format_fact(fact):
    """
    Formats a fact dictionary into a string.

    Args:
        fact (dict): A dictionary containing 'subject', 'verb', 'direct_object', and 'prepositional_objects'.

    Returns:
        str: A formatted string representing the fact.
    """
    subject = fact.get("subject", "")
    verb = fact.get("verb", "")
    direct_object = fact.get("direct_object", "")
    prepositional_objects = fact.get("prepositional_objects", [])

    fact_str = f"{subject} {verb}"
    if direct_object:
        fact_str += f" {direct_object}"
    if prepositional_objects:
        fact_str += f" {' '.join(prepositional_objects)}"

    return fact_str.strip()