# decomposition/decomposition_module.py

import spacy
from spacy.tokens import Token
import coreferee

# Initialize SpaCy with coreferee
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('coreferee')

def resolve_coreferences(doc):
    """
    Resolve coreferences in the document using coreferee.

    Args:
        doc (spacy.Doc): The SpaCy document.

    Returns:
        str: The text with pronouns replaced by their antecedents.
    """
    # Check if coreference resolution is needed
    if not doc._.has_coref or not doc._.coref_chains:
        return doc.text

    token_replacements = {}

    for chain in doc._.coref_chains:
        main = chain.main
        for mention in chain:
            if mention == chain.main:
                continue
            for token in mention:
                if token.pos_ == "PRON":
                    token_replacements[token] = main

    resolved_tokens = []
    for token in doc:
        if token in token_replacements:
            replacement = token_replacements[token]
            resolved_tokens.append(token_replaced(token, replacement.text))
        else:
            resolved_tokens.append(token.text)

    # Reconstruct the resolved text
    resolved_text = ""
    for i, token in enumerate(doc):
        if resolved_tokens[i] in ["'", "’"] and i > 0:
            resolved_text = resolved_text.rstrip() + resolved_tokens[i] + " "
        else:
            resolved_text += resolved_tokens[i] + " "
    return resolved_text.strip()

def token_replaced(token, replacement_text):
    """
    Handle capitalization and possessive forms when replacing tokens.

    Args:
        token (spacy.Token): The original token.
        replacement_text (str): The replacement text.

    Returns:
        str: The appropriately formatted replacement text.
    """
    if token.text.istitle():
        return replacement_text.capitalize()
    elif token.text.isupper():
        return replacement_text.upper()
    else:
        return replacement_text

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

def decompose_sentence(sentence):
    """
    Decompose a sentence into atomic facts by extracting comprehensive subject-verb-object relationships,
    including various object types and handling complex sentence structures.

    Args:
        sentence (str): The sentence to decompose.

    Returns:
        list: A list of dictionaries, each containing 'subject', 'verb', 'direct_object', and 'prepositional_objects'.
              Returns an empty list if decomposition isn't possible.
    """
    try:
        doc = nlp(sentence)
        resolved_text = resolve_coreferences(doc)
        resolved_doc = nlp(resolved_text)
    except Exception as e:
        print(f"Error parsing sentence: {e}")
        return []

    facts = []
    current_subject = None  # To hold the subject from the previous clause

    for sent in resolved_doc.sents:
        # Check if the sentence has a main verb with ROOT dependency
        has_root = any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in sent)
        if not has_root:
            # No main verb; skip this sentence
            print(f"Skipping sentence with no main verb: {sent.text}")
            continue

        # Iterate through tokens to find verbs
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"):
                verb = token.text  # Preserve original verb form
                verb_aux = ''

                # Include auxiliary verbs (e.g., is, was, have been)
                aux = [child.text for child in token.children if child.dep_ in ('aux', 'auxpass')]
                if aux:
                    verb_aux = ' '.join(aux)
                    full_verb = f"{verb_aux} {verb}"
                else:
                    full_verb = verb

                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        subject = child.text
                        # Include compounds and appositives in subject
                        subject_modifiers = [w.text for w in child.lefts if w.dep_ in ('det', 'amod', 'compound', 'appos')]
                        if subject_modifiers:
                            subject = ' '.join(subject_modifiers + [subject])
                        break
                if not subject:
                    subject = current_subject  # Inherit subject from previous clause

                # Update current_subject if a new subject is found
                if subject:
                    current_subject = subject

                # Find direct objects
                direct_objects = []
                for child in token.children:
                    if child.dep_ in ('dobj', 'attr', 'oprd', 'iobj'):
                        obj = get_full_object(child)
                        direct_objects.append(obj)

                # Find prepositional objects
                prepositional_objects = []
                for child in token.children:
                    if child.dep_ == 'prep':
                        for pobj in child.children:
                            if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON'):
                                prep_obj = get_full_object(pobj)
                                prepositional_objects.append(f"{child.text.capitalize()} {prep_obj}")

                # Handle relative clauses attached to the main verb's object
                for child in token.children:
                    if child.dep_ == 'relcl':
                        # Extract relative clause as a separate fact
                        relative_verb = child.text
                        relative_subject = None
                        # Find subject of the relative clause
                        for rel_child in child.children:
                            if rel_child.dep_ in ('nsubj', 'nsubjpass'):
                                relative_subject = rel_child.text
                                # Include modifiers
                                rel_modifiers = [w.text for w in rel_child.lefts if w.dep_ in ('det', 'amod', 'compound', 'appos')]
                                if rel_modifiers:
                                    relative_subject = ' '.join(rel_modifiers + [relative_subject])
                                break
                        if relative_subject:
                            # Find direct objects and prepositional objects in the relative clause
                            relative_direct_objects = []
                            relative_prepositional_objects = []
                            for rel_child in child.children:
                                if rel_child.dep_ in ('dobj', 'attr', 'oprd', 'iobj'):
                                    obj = get_full_object(rel_child)
                                    relative_direct_objects.append(obj)
                                elif rel_child.dep_ == 'prep':
                                    for rel_pobj in rel_child.children:
                                        if rel_pobj.dep_ == 'pobj' and rel_pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON'):
                                            prep_obj = get_full_object(rel_pobj)
                                            relative_prepositional_objects.append(f"{rel_child.text.capitalize()} {prep_obj}")
                            # Add relative clause fact
                            relative_fact = {
                                "subject": relative_subject.strip(','),
                                "verb": relative_verb.lower(),
                                "direct_object": ', '.join(relative_direct_objects).capitalize() if relative_direct_objects else "",
                                "prepositional_objects": relative_prepositional_objects if relative_prepositional_objects else []
                            }
                            if relative_fact["subject"] and relative_fact["verb"]:
                                facts.append(relative_fact)

                # Add main verb fact
                if subject and full_verb:
                    fact = {
                        "subject": subject.strip(','),  # Remove trailing commas
                        "verb": full_verb.lower(),
                        "direct_object": ', '.join(direct_objects).capitalize() if direct_objects else "",
                        "prepositional_objects": prepositional_objects if prepositional_objects else []
                    }
                    # Ensure atomicity: at least subject and verb
                    if fact["subject"] and fact["verb"]:
                        facts.append(fact)

                # Handle conjunct verbs sharing the same subject
                for conj in token.conjuncts:
                    conj_verb = conj.text
                    conj_aux = ''
                    # Include auxiliary verbs for conjunct verbs
                    aux_conj = [child.text for child in conj.children if child.dep_ in ('aux', 'auxpass')]
                    if aux_conj:
                        conj_aux = ' '.join(aux_conj)
                        full_conj_verb = f"{conj_aux} {conj_verb}"
                    else:
                        full_conj_verb = conj_verb

                    # Find direct objects for conjunct verbs
                    conj_direct_objects = []
                    for child in conj.children:
                        if child.dep_ in ('dobj', 'attr', 'oprd', 'iobj'):
                            obj = get_full_object(child)
                            conj_direct_objects.append(obj)

                    # Find prepositional objects for conjunct verbs
                    conj_prepositional_objects = []
                    for child in conj.children:
                        if child.dep_ == 'prep':
                            for pobj in child.children:
                                if pobj.dep_ == 'pobj' and pobj.pos_ in ('NOUN', 'PROPN', 'ADJ', 'PRON'):
                                    prep_obj = get_full_object(pobj)
                                    conj_prepositional_objects.append(f"{child.text.capitalize()} {prep_obj}")

                    # Add conjunct verb fact
                    if subject and full_conj_verb:
                        conj_fact = {
                            "subject": subject.strip(','),
                            "verb": full_conj_verb.lower(),
                            "direct_object": ', '.join(conj_direct_objects).capitalize() if conj_direct_objects else "",
                            "prepositional_objects": conj_prepositional_objects if conj_prepositional_objects else []
                        }
                        if conj_fact["subject"] and conj_fact["verb"]:
                            facts.append(conj_fact)

        # Additional post-processing for atomicity
        # Split facts where the direct_object or prepositional_objects contain multiple pieces of information
        atomic_facts = []
        for fact in facts:
            # Check direct_object for multiple pieces
            direct_objects = [obj.strip() for obj in fact.get('direct_object', '').split(',') if obj.strip()]
            if len(direct_objects) > 1:
                for obj in direct_objects:
                    atomic_fact = fact.copy()
                    atomic_fact['direct_object'] = obj
                    atomic_facts.append(atomic_fact)
            else:
                # Check prepositional_objects for multiple pieces
                preps = fact.get('prepositional_objects', [])
                if len(preps) > 1:
                    for prep in preps:
                        atomic_fact = fact.copy()
                        atomic_fact['prepositional_objects'] = [prep]
                        atomic_facts.append(atomic_fact)
                else:
                    atomic_facts.append(fact)

        # Final atomicity check: ensure each fact is atomic
        final_facts = []
        for fact in atomic_facts:
            if fact["subject"] and fact["verb"]:
                final_facts.append(fact)

        return final_facts
