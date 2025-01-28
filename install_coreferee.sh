#!/bin/bash

# Install coreferee
pip install --no-cache-dir coreferee

# Add coreferee to the SpaCy pipeline
python -m spacy add-pipe coreferee en_core_web_sm

# Continue with any other commands or start your application
exec "$@"