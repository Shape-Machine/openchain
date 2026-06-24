from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown
from tools.outreach import get_budget_allocations, get_segments, set_budget_allocations
from tools.sales import get_sales_actuals, get_sales_targets

MODEL = "anthropic:claude-sonnet-4-6"

SYSTEM_PROMPT = """
Our mission is to allocate budget for outreach activities across various segments.

For this, we review the sales, targets, and actuals.

And then adjust outreach budget accordingly.
"""

TOOLS = [
    get_segments,
    get_budget_allocations,
    set_budget_allocations,
    get_sales_actuals,
    get_sales_targets,
]


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

"""
Fetch all customers
    - For each customer:
        - Low usage
            - New customer
            - Old customer
        - Mid usage
            - Newish customer
            - Old customer
        - High usage
            - Exceeds limits
            - Within but close to limits
            - Distant from limits
    - Compile report
    - Suggested actions

Tools:
    - CRM
    - UsageData

"""
