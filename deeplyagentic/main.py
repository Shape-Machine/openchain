import os

from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown
from tools.outreach import get_budget_allocations, get_segments, set_budget_allocations
from tools.sales import get_sales_actuals, get_sales_targets

os.environ.setdefault("LANGSMITH_PROJECT", "OpenChain")

# google_genai:gemini-3.5-flash
MODEL = "anthropic:claude-sonnet-4-6"

SYSTEM_PROMPT = """
Fetch all customers and for each customer:

    1. Analyize usage and customer age
    2. If:
        - the usage is low and they are a new customer
        - the usage is low and they are an old customer
        - the usage is mid and they are a newish customer
        - the usage is mid and they are an old customer
        - the usage is high and they are within but close to limits
        - the usage is high and they are distant from limits
        - the usage is high and they exceed limits
    3. Compile report
    4. Suggest customer success actions:
        - Onboard
        - Highlight capability
        - Highlight value
        - Upsell
        - Handle overage
"""

SUBAGENTS = []

TOOLS = [
    # CRM
    # DataWarehouse::Usage
    get_segments,
    get_budget_allocations,
    set_budget_allocations,
    get_sales_actuals,
    get_sales_targets,
]

SUBAGENTS = []


def main():
    agent = create_deep_agent(
        system_prompt=SYSTEM_PROMPT, model=MODEL, tools=TOOLS, memory=[], skills=[]
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Review the current sales actuals and the targets, "
                    "and then review the budget and reallocate the budget across all segments. "
                    "Then give me a summary at the end explaining what's the current state. ",
                }
            ]
        }
    )

    Console().print(Markdown(result["messages"][-1].content))


if __name__ == "__main__":
    main()
