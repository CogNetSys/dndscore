# agents/WebSearchAgent.py

import os
import requests
import json
from typing import List, Dict
from smolagents import Tool, tool
import time

# Define the DuckDuckGoSearchTool
@tool
def duckduckgo_search(query: str, num_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo and returns the search results in JSON format.

    Args:
        query (str): The search query.
        num_results (int): The number of search results to return.

    Returns:
        str: A JSON string containing search results with 'title', 'snippet', and 'link'.
    """
    api_url = "https://api.duckduckgo.com/"
    params = {
        'q': query,
        'format': 'json',
        'no_html': 1,
        'skip_disambig': 1
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        # DuckDuckGo's Instant Answer API provides 'RelatedTopics' as search results
        for topic in data.get('RelatedTopics', []):
            if 'Text' in topic and 'FirstURL' in topic:
                results.append({
                    'title': topic.get('Name', 'No Title'),
                    'snippet': topic.get('Text', ''),
                    'link': topic.get('FirstURL', '')
                })
                if len(results) >= num_results:
                    break
            elif 'Topics' in topic:
                for subtopic in topic.get('Topics', []):
                    if 'Text' in subtopic and 'FirstURL' in subtopic:
                        results.append({
                            'title': subtopic.get('Name', 'No Title'),
                            'snippet': subtopic.get('Text', ''),
                            'link': subtopic.get('FirstURL', '')
                        })
                        if len(results) >= num_results:
                            break
        return json.dumps(results)
    except requests.exceptions.RequestException as e:
        return json.dumps([{'title': 'Error', 'snippet': str(e), 'link': ''}])
    except Exception as e:
        return json.dumps([{'title': 'Unexpected Error', 'snippet': str(e), 'link': ''}])

# Define the VisitWebpageTool
@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url (str): The URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    import re
    from markdownify import markdownify
    from requests.exceptions import RequestException

    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
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

# Initialize the WebSearchAgent
def initialize_web_search_agent(model, max_steps: int = 5, additional_imports: List[str] = []):
    """
    Initializes the WebSearchAgent with necessary tools.

    Args:
        model: The language model to use.
        max_steps (int): Maximum number of steps the agent can take.
        additional_imports (List[str]): Additional modules to authorize.

    Returns:
        ManagedAgent: The initialized WebSearchAgent.
    """
    from smolagents import ToolCallingAgent, ManagedAgent

    # Create the search agent with DuckDuckGoSearchTool and VisitWebpageTool
    search_agent = ToolCallingAgent(
        tools=[duckduckgo_search, visit_webpage],
        model=model,
        max_steps=max_steps,
    )

    # Create the managed web search agent
    managed_search_agent = ManagedAgent(
        agent=search_agent,
        name="web_searcher",
        description="Performs web searches and can visit web pages to extract information. Provide a detailed query.",
    )

    return managed_search_agent
