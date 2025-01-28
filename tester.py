import sys
from transformers import pipeline

def main():
    print("Verifying Hugging Face Transformers installation...")

    try:
        # Load the coreference resolution pipeline with a pre-trained model
        print("Loading coreference resolution pipeline...")
        nlp = pipeline("coreference-resolution", model="facebook/bart-large-mnli")

        # Test example text
        text = "John took his dog to the park. He loves the park."
        print(f"Input text: {text}")

        # Perform coreference resolution
        print("Running coreference resolution...")
        result = nlp(text)

        # Output results
        print("\nCoreference Resolution Result:")
        print(result)

        print("\nInstallation and functionality verified successfully!")

    except Exception as e:
        print("\nAn error occurred during the test.")
        print(f"Error details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
