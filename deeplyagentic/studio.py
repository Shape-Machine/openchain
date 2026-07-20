"""LangSmith Studio entry point for the RiffDesk agent.

LangGraph Agent Server injects checkpointing and exposes interrupts in Studio,
so this adapter deliberately contains no terminal UI or local checkpointer.
"""

import json

from langchain.tools import tool
from langgraph.types import interrupt

from deeplyagentic.core import build_agent, request_refund


def _parse_studio_resume(response: object) -> dict:
    """Normalize Studio's JSON-string resume values to native mappings."""
    if isinstance(response, str):
        response = json.loads(response)
    if not isinstance(response, dict):
        raise ValueError("Studio resume value must be a JSON object")
    return response


@tool("request_refund")
def request_refund_with_studio_approval(
    customer_id: int,
    invoice_id: int,
    reason: str,
) -> str:
    """File a refund request for a verified customer's invoice after approval."""
    action = {
        "name": "request_refund",
        "args": {
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "reason": reason,
        },
        "description": (
            "Tool execution requires approval\n\n"
            f"Tool: request_refund\nArgs: customer_id={customer_id}, "
            f"invoice_id={invoice_id}, reason={reason!r}"
        ),
    }
    response = _parse_studio_resume(
        interrupt(
            {
                "action_requests": [action],
                "review_configs": [
                    {
                        "action_name": "request_refund",
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            }
        )
    )
    decisions = response.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Studio resume value must contain exactly one decision")

    decision = decisions[0]
    if not isinstance(decision, dict):
        raise ValueError("Studio decision must be a JSON object")
    if decision.get("type") == "reject":
        return "The customer rejected the refund request; no refund was filed."
    if decision.get("type") != "approve":
        raise ValueError("Studio decision type must be 'approve' or 'reject'")

    return request_refund.invoke(
        {
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "reason": reason,
        }
    )


agent = build_agent(
    refund_tool=request_refund_with_studio_approval,
    require_refund_approval=False,
)
