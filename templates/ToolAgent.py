#!/usr/bin/env python
# coding=utf-8
from smolagents import ToolCallingAgent, LiteLLMModel, Tool
from dotenv import load_dotenv

load_dotenv()


# Define a sample tool (you can replace this with your own tools)
class GreetingTool(Tool):
    name = "greeter"
    description = "Greets the person with the given name."
    inputs = {
        "name": {"type": "string", "description": "The name of the person to greet."}
    }
    output_type = "string"

    def forward(self, name: str) -> str:
        return f"Hello, {name}! It's nice to see you."


# Initialize the Groq model
your_groq_model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# Create tool instances
greeting_tool = GreetingTool()

# Define a regex grammar for the tool-calling output
json_grammar = {
    "type": "regex",
    "value": r'\{\s*"action":\s*"[\w_]+",\s*"action_input":\s*\{.*\}\s*\}',
}

# Initialize the ToolCallingAgent
tool_calling_agent = ToolCallingAgent(
    model=your_groq_model,
    tools=[greeting_tool],
    grammar=json_grammar,
    max_steps=3,  # You can adjust this
)

# Run the agent
result = tool_calling_agent.run(
    "Could you greet my friend Don using the appropriate tool?"
)
print(result)
