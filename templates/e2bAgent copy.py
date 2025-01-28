import os
from smolagents import CodeAgent, LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

# Initialize the Groq model using LiteLLM
your_groq_model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# Initialize the CodeAgent with E2B enabled
code_agent = CodeAgent(
    model=your_groq_model,
    tools=[],  # Start with no tools for now
    additional_authorized_imports=[],  # No extra imports
    use_e2b_executor=True,
)

# Run a very simple test
try:
    result = code_agent.run("What is one plus one?")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
