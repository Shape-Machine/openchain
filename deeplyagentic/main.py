"""
RiffDesk support supervisor and specialist agents. Their tools call the
riffdesk REST API (riffdesk/main.py) instead of touching SQLite directly.

Identity is now collected via a real LangGraph interrupt: the customer_id
and email are NOT in the initial prompt. The collect_customer_identity tool
pauses the graph mid-execution and asks for them directly.

Run riffdesk first:  uv run riffdesk/main.py
Then run this module:  uv run python -m deeplyagentic.main

customer id: 1
email: luisg@embraer.com.br
"""

from uuid import uuid4

import questionary
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from prompt_toolkit.validation import ValidationError as PromptValidationError
from prompt_toolkit.validation import Validator
from pydantic import BaseModel, EmailStr, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from deeplyagentic.api import api_client
from deeplyagentic.subagents import purchase_specialist, refund_specialist

console = Console()


# --- Validation schema for the identity-collection interrupt ---
# This is the authoritative check: whatever a UI does upstream, the tool
# itself never trusts raw resume data.
class IdentityInput(BaseModel):
    customer_id: int
    email: EmailStr


# --- Fast, client-side checks for the prompt itself (UX layer only) ---
# These don't replace IdentityInput above -- they just stop obviously bad
# input from being submitted in the first place. The tool still validates
# authoritatively regardless of what happens here.
class _CustomerIdPromptValidator(Validator):
    def validate(self, document):
        text = document.text.strip()
        if not text.isdigit():
            raise PromptValidationError(
                message="Customer ID must be a number",
                cursor_position=len(text),
            )


class _EmailPromptValidator(Validator):
    def validate(self, document):
        text = document.text.strip()
        if "@" not in text or "." not in text.split("@")[-1]:
            raise PromptValidationError(
                message="Please enter a valid-looking email address",
                cursor_position=len(text),
            )


# --- Tool 1: identity collection via a direct interrupt() ---
@tool
def collect_customer_identity() -> str:
    """Call this FIRST, before any other tool. Pauses and asks the customer
    for their customer ID and email, then verifies identity. Returns the
    verified customer_id to use in later tool calls."""
    error = None

    # Loop until we get a structurally valid customer_id + email. Each
    # iteration is its own interrupt/resume — the graph pauses again with
    # the error attached instead of ever raising or crashing.
    while True:
        payload = {
            "type": "identity_request",
            "message": "Please provide your customer ID and email to verify your identity.",
        }
        if error:
            payload["error"] = error

        response = interrupt(payload)

        try:
            identity = IdentityInput(**response)
            break
        except ValidationError as exc:
            # Turn pydantic's error into one plain-English line per field.
            error = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())

    with api_client() as client:
        resp = client.get(f"/customers/{identity.customer_id}")
    if resp.status_code == 404:
        return f"No customer found with ID {identity.customer_id}. Ask the customer to try again."
    resp.raise_for_status()
    customer = resp.json()

    if customer["email"].strip().lower() != identity.email.strip().lower():
        return "Identity verification FAILED: email does not match our records. Do not proceed."

    return (
        f"Identity verified for {customer['first_name']} {customer['last_name']} "
        f"(customer_id={identity.customer_id}). Use this customer_id for all further lookups."
    )


# --- Wire up the supervisor and specialists ---
checkpointer = MemorySaver()  # required for human-in-the-loop

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[collect_customer_identity],
    subagents=[purchase_specialist, refund_specialist],
    system_prompt=(
        "You are the customer-facing support supervisor for RiffDesk. "
        "Always call collect_customer_identity FIRST, before anything else, "
        "and do not delegate until it succeeds. Pass the verified customer_id "
        "and all relevant customer context in every specialist task. Delegate "
        "purchase-history and invoice-detail work to purchase-specialist. "
        "Delegate refunds to refund-specialist. Do not attempt specialist work "
        "yourself, and never substitute a different customer_id. Present each "
        "specialist's result clearly to the customer."
    ),
    checkpointer=checkpointer,
)


def _prompt_decision(action: dict) -> str:
    """A select menu instead of free text — approve/reject can't be typed
    wrong because there's nothing to type. Shortcut keys 'a'/'r' still work
    for a fast keyboard-only flow."""
    console.print(
        Panel(
            f"[bold]{action['name']}[/bold]({action['args']})",
            title="Approval needed",
            border_style="yellow",
        )
    )
    return questionary.select(
        "How do you want to handle this?",
        choices=[
            questionary.Choice("Approve", value="approve", shortcut_key="a"),
            questionary.Choice("Reject", value="reject", shortcut_key="r"),
        ],
        use_shortcuts=True,
    ).ask()


def _handle_interrupt_loop(result, config):
    """Drive the demo interactively: whenever the graph pauses, figure out
    which kind of interrupt it is and ask a human for the missing piece via
    Rich/questionary, then resume. Nothing here ever raises on bad input —
    the worst case is asking again."""
    while result.interrupts:
        value = result.interrupts[0].value

        if value.get("type") == "identity_request":
            if value.get("error"):
                console.print(f"[bold red]Error:[/bold red] {value['error']}")
            console.print(Panel(value["message"], title="Identity verification", border_style="cyan"))

            # Validators here are the fast UX layer only — IdentityInput
            # inside the tool is still the authoritative check and can
            # re-interrupt even if these somehow get bypassed.
            customer_id = questionary.text(
                "Customer ID:", validate=_CustomerIdPromptValidator
            ).ask()
            email = questionary.text("Email:", validate=_EmailPromptValidator).ask()

            result = agent.invoke(
                Command(resume={"customer_id": customer_id, "email": email}),
                config=config,
                version="v2",
            )

        elif "action_requests" in value:
            action = value["action_requests"][0]
            decision = _prompt_decision(action)
            result = agent.invoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
                version="v2",
            )

        else:
            raise RuntimeError(f"Unhandled interrupt payload: {value}")

    return result


if __name__ == "__main__":
    config = {"configurable": {"thread_id": f"demo-{uuid4()}"}}

    console.print(Panel("RiffDesk support bot — type 'exit' to quit.", style="bold green"))

    # Note: no customer_id or email in the initial prompt on purpose.
    user_input = (
        "Hi, what did I buy recently? Also I'd like a refund on invoice 12, wrong track."
    )

    while True:
        with console.status("[bold cyan]Agent is thinking...[/bold cyan]"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2",
            )

        result = _handle_interrupt_loop(result, config)

        console.print(Panel(Markdown(result.value["messages"][-1].content), title="Agent", border_style="blue"))

        user_input = questionary.text("You:").ask()
        if user_input is None or user_input.strip().lower() in {"exit", "quit"}:
            break
