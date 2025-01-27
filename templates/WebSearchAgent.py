from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    Tool,
    tool
)
from dotenv import load_dotenv
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException

load_dotenv()

# Define the VisitWebpageTool
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Initialize the Groq model (replace with your preferred model)
# model = LiteLLMModel(model_id="groq/llama3-70b-8192")
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Create the web search agent
web_search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=5,
)

# Create the managed web search agent
managed_web_search_agent = ManagedAgent(
    agent=web_search_agent,
    name="web_searcher",
    description="Performs web searches and can visit web pages to extract information. Provide a detailed query.",
)

# Create the manager agent
manager_agent = CodeAgent(
    model=model,
    managed_agents=[managed_web_search_agent],
    max_steps=7,
    additional_authorized_imports=["time", "numpy", "pandas"],
    use_e2b_executor=False,
    tools=[]
)

# Run the manager agent
task = "What is the current population of America? Please use web search to find the latest information."
result = manager_agent.run(task)
print(result)