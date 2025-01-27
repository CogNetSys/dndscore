```
dndscore/
│
├── decomposition/        # DR-ND logic (ChatGPT's current focus)
│   └── decomposition_module.py # Contains the decompose_sentence function
├── decontextualization/  # Molecular Facts prompts and potentially a decontextualization module
│   └── decontext_module.py # Contains the decontextualize function
│   └── molecular_facts_prompts.py # Prompt templates for decontextualization
├── verification/         # Entailment models and verification logic
│   └── verifier.py      # Contains the dndscore_verify function
├── core/                 # Deduplication modules (CORE implementation)
│   └── core_module.py    # Contains the CORE class and methods
├── prompts/              # Folder for various prompt files
│   └── dnd_score_prompts.py # Prompt templates for DnDScore
│   └── factscore_decomposition_prompt.txt # Prompt for decomposition used in FACTSCORE
├── pipeline.py           # Integrated workflow
├── test_sentences.txt    # Test sentences for decomposition
├── README.md             # Project description
└── requirements.txt      # Python dependencies
```


Workflow for Running Your Project
1. Start the Docker Container:

```bash
docker run -it --rm -v $(pwd):/app coreferee-env bash
```

2. Navigate to /app:

```bash
cd /app
```

3. Activate the venv inside the container:

```bash
source venv/bin/activate
```

4. Run your scripts or programs:

```bash
python your_script.py
```
