import re
import os
from typing import Callable, List
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool

def __init__(self, tools: List[Callable]):
    print("hi")
    
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url (str): The URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown, or an error message if the request fails.
    """

    print('web search tool invoked. I will try visit weather.com to fetch weather for the requested cities')

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
    