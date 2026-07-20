"""
Agent tools for the RiffDesk support bot — these call the riffdesk REST API
(riffdesk/main.py) over HTTP instead of touching SQLite directly.

Identity is now collected via a real LangGraph interrupt: the customer_id
and email are NOT in the initial prompt. The collect_customer_identity tool
pauses the graph mid-execution and asks for them directly.

Run riffdesk first:  uv run riffdesk/main.py
Then run this file:  uv run agent_tools.py

customer id: 1
email: luisg@embraer.com.br

{"customer_id": 1, "email": "luisg@embraer.com.br"}
{
   "decisions": [
     {"type": "approve"}
   ]
 }
"""

import questionary
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from prompt_toolkit.validation import ValidationError as PromptValidationError
from prompt_toolkit.validation import Validator
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .core import build_agent

console = Console()

# A restrained semantic palette: blue is RiffDesk chrome, cyan indicates an
# active interaction, yellow is reserved for consequential actions, and red
# is reserved for errors.
APP_COLOR = "bright_blue"
INTERACTION_COLOR = "cyan"
CAUTION_COLOR = "yellow"
ERROR_STYLE = "bold red"

PROMPT_STYLE = questionary.Style(
    [
        ("qmark", f"fg:{INTERACTION_COLOR} bold"),
        ("question", "bold"),
        ("answer", f"fg:{INTERACTION_COLOR} bold"),
        ("pointer", f"fg:{INTERACTION_COLOR} bold"),
        ("highlighted", f"fg:{INTERACTION_COLOR} bold"),
        ("selected", f"fg:{INTERACTION_COLOR}"),
        ("instruction", "fg:#888888"),
        ("text", ""),
        ("disabled", "fg:#888888 italic"),
    ]
)


# --- Fast, client-side checks for the prompt itself (UX layer only) ---
# These don't replace IdentityInput in core.py -- they just stop obviously
# bad input from being submitted in the first place. The tool still validates
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


# The terminal variant owns its in-process persistence. Agent Server supplies
# persistence separately for the Studio graph exported from studio.py.
agent = build_agent(checkpointer=MemorySaver())


def _prompt_decision(action: dict) -> str:
    """A select menu instead of free text — approve/reject can't be typed
    wrong because there's nothing to type. Shortcut keys 'a'/'r' still work
    for a fast keyboard-only flow."""
    console.print(
        Panel(
            f"[bold]{action['name']}[/bold]({action['args']})",
            title="Approval needed",
            border_style=CAUTION_COLOR,
        )
    )
    return questionary.select(
        "How do you want to handle this?",
        choices=[
            questionary.Choice("Approve", value="approve", shortcut_key="a"),
            questionary.Choice("Reject", value="reject", shortcut_key="r"),
        ],
        use_shortcuts=True,
        style=PROMPT_STYLE,
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
                console.print(f"[{ERROR_STYLE}]Error:[/{ERROR_STYLE}] {value['error']}")
            console.print(
                Panel(
                    value["message"],
                    title="Identity verification",
                    border_style=APP_COLOR,
                )
            )

            # Validators here are the fast UX layer only — IdentityInput
            # inside the tool is still the authoritative check and can
            # re-interrupt even if these somehow get bypassed.
            customer_id = questionary.text(
                "Customer ID:",
                validate=_CustomerIdPromptValidator,
                style=PROMPT_STYLE,
            ).ask()
            email = questionary.text(
                "Email:", validate=_EmailPromptValidator, style=PROMPT_STYLE
            ).ask()

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
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    console.print(
        Panel(
            "RiffDesk support bot — type 'exit' to quit.",
            border_style=APP_COLOR,
        )
    )

    # Note: no customer_id or email in the initial prompt on purpose.
    user_input = (
        "Hi, what did I buy recently? Also I'd like a refund on invoice 12, wrong track."
    )

    while True:
        with console.status(
            f"[bold {INTERACTION_COLOR}]Agent is thinking...[/bold {INTERACTION_COLOR}]"
        ):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2",
            )

        result = _handle_interrupt_loop(result, config)

        console.print(
            Panel(
                Markdown(result.value["messages"][-1].content),
                title="Agent",
                border_style=APP_COLOR,
            )
        )

        user_input = questionary.text("You:", style=PROMPT_STYLE).ask()
        if user_input is None or user_input.strip().lower() in {"exit", "quit"}:
            break
