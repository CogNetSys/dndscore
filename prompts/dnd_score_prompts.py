DND_SCORE_PROMPT = """
Decompose the following sentence into a list of subclaims.
Then, for each subclaim, decontextualize it to be a standalone claim using the given context.
For each decontextualized claim, include additional context from the context to the decontextualized claim to help verify.

Context:
{context}

Sentence:
{sentence}

Output a list of (subclaim, decontextualized, context) tuples.

Example:

Context:
Al Pacino is an American actor. He was born in New York City, on April 25, 1940. Pacino's parents divorced when he was two. He moved with his mother to live with her parents. Pacino attended the High School of Performing Arts. He left school at age 17.

Sentence:
He was born in New York City, on April 25, 1940.

Subclaims:
- He was born in New York City.
- He was born on April 25, 1940.

Decontextualized Claims:
- Al Pacino was born in New York City.
- Al Pacino was born on April 25, 1940.

Context:
- Al Pacino is an American actor. He was born in New York City, on April 25, 1940.
- Al Pacino is an American actor. He was born in New York City, on April 25, 1940.

Output:
[("He was born in New York City", "Al Pacino was born in New York City", "Al Pacino is an American actor. He was born in New York City, on April 25, 1940."), ("He was born on April 25, 1940", "Al Pacino was born on April 25, 1940", "Al Pacino is an American actor. He was born in New York City, on April 25, 1940.")]
"""