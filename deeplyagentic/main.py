"""
Agent tools for the RiffDesk support bot — these call the riffdesk REST API
(riffdesk/main.py) over HTTP instead of touching SQLite directly.

Identity is now collected via a real LangGraph interrupt: the customer_id
and email are NOT in the initial prompt. The collect_customer_identity tool
pauses the graph mid-execution and asks for them directly.

Run riffdesk first:  uv run riffdesk/main.py
Then run this file:  uv run agent_tools.py
"""

import os

import httpx
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

API_BASE_URL = os.environ.get("RIFFDESK_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("RIFFDESK_API_KEY", "demo-secret-key")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10.0,
    )


# --- Tool 1: identity collection via a direct interrupt() ---
@tool
def collect_customer_identity() -> str:
    """Call this FIRST, before any other tool. Pauses and asks the customer
    for their customer ID and email, then verifies identity. Returns the
    verified customer_id to use in later tool calls."""
    # This pauses the graph right here. Whatever dict is passed to
    # Command(resume=...) on resumption becomes the return value below.
    response = interrupt(
        {
            "type": "identity_request",
            "message": "Please provide your customer ID and email to verify your identity.",
        }
    )
    customer_id = response["customer_id"]
    email = response["email"]

    with _client() as client:
        resp = client.get(f"/customers/{customer_id}")
    if resp.status_code == 404:
        return f"No customer found with ID {customer_id}. Ask the customer to try again."
    resp.raise_for_status()
    customer = resp.json()

    if customer["email"].strip().lower() != email.strip().lower():
        return "Identity verification FAILED: email does not match our records. Do not proceed."

    return (
        f"Identity verified for {customer['first_name']} {customer['last_name']} "
        f"(customer_id={customer_id}). Use this customer_id for all further lookups."
    )


# --- Tool 2: order/invoice lookup (read-only) ---
@tool
def query_invoices(customer_id: int, limit: int = 10) -> str:
    """Look up a verified customer's recent invoices (id, date, total)."""
    with _client() as client:
        resp = client.get(f"/customers/{customer_id}/invoices", params={"limit": limit})
    if resp.status_code == 404:
        return f"No customer found with ID {customer_id}."
    resp.raise_for_status()
    return str(resp.json())


# --- Tool 3: invoice line-item detail ---
@tool
def get_invoice_detail(invoice_id: int) -> str:
    """Get line-item detail (tracks, price, quantity) for a specific invoice.
    Use this to confirm exactly what's being refunded before filing a refund."""
    with _client() as client:
        resp = client.get(f"/invoices/{invoice_id}")
    if resp.status_code == 404:
        return f"No invoice found with ID {invoice_id}."
    resp.raise_for_status()
    return str(resp.json())


# --- Tool 4: refund request (write, sensitive -> gated via interrupt_on) ---
@tool
def request_refund(invoice_id: int, reason: str) -> str:
    """File a refund request for a given invoice. Requires human approval
    before it takes effect."""
    with _client() as client:
        resp = client.post("/refunds", json={"invoice_id": invoice_id, "reason": reason})
    if resp.status_code == 404:
        return f"No invoice found with ID {invoice_id}; refund not filed."
    resp.raise_for_status()
    return str(resp.json())


# --- Wire up the agent ---
checkpointer = MemorySaver()  # required for human-in-the-loop

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[collect_customer_identity, query_invoices, get_invoice_detail, request_refund],
    system_prompt=(
        "You are a music store support agent for RiffDesk. "
        "Always call collect_customer_identity FIRST, before anything else, "
        "and only proceed to query_invoices, get_invoice_detail, or "
        "request_refund once identity has been verified successfully. "
        "Confirm the exact invoice with the customer before filing a refund."
    ),
    interrupt_on={
        # collect_customer_identity is NOT listed here — it pauses via its
        # own direct interrupt() call, independent of interrupt_on.
        "query_invoices": False,
        "get_invoice_detail": False,
        "request_refund": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=checkpointer,
)


def _handle_interrupt_loop(result, config):
    """Drive the demo interactively: whenever the graph pauses, figure out
    which kind of interrupt it is and ask a human (via input()) for the
    missing piece, then resume."""
    while result.interrupts:
        value = result.interrupts[0].value

        if value.get("type") == "identity_request":
            print(f"\n[agent] {value['message']}")
            customer_id = int(input("  customer_id: ").strip())
            email = input("  email: ").strip()
            result = agent.invoke(
                Command(resume={"customer_id": customer_id, "email": email}),
                config=config,
                version="v2",
            )

        elif "action_requests" in value:
            action = value["action_requests"][0]
            print(f"\n[approval needed] {action['name']}({action['args']})")
            decision = input("  approve or reject: ").strip().lower()
            result = agent.invoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
                version="v2",
            )

        else:
            raise RuntimeError(f"Unhandled interrupt payload: {value}")

    return result


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    print("RiffDesk support bot — type 'exit' to quit.\n")
    user_input = (
        "Hi, what did I buy recently? Also I'd like a refund on invoice 12, wrong track."
    )

    while True:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            version="v2",
        )

        result = _handle_interrupt_loop(result, config)

        print(f"\n[agent] {result.value['messages'][-1].content}\n")

        user_input = input("[you] ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
