FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /app

# Install additional Python dependencies (if needed)
RUN pip install --no-cache-dir \
    allennlp \
    allennlp-models

# Copy your application code
COPY . .

# Set the default command to run your test script
CMD ["python", "tester.py"]
