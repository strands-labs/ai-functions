import os

def get_websearch_tool():
    """Search environment variables for an API key of the supported websearch tools and return the corresponding tool."""
    # see if we have API keys for any of the websearch tools supported by strands_tools
    if os.environ.get("EXA_API_KEY"):
        from strands_tools import exa as websearch_tool
    elif os.environ.get("TAVILY_API_KEY"):
        from strands_tools import tavily as websearch_tool
    else:
        raise ValueError("You need to set an environment variable containing the API key"
                         "for exa (EXA_API_KEY) or tavily (TAVILY_API_KEY)")
    return websearch_tool