from __future__ import annotations

from dataclasses import dataclass

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import OptionList
from textual.widgets.option_list import Option


@dataclass
class SelectItem:
    id: str
    label: str
    description: str | None = None


class SelectScreen(ModalScreen[SelectItem | None]):
    """A modal, arrow-key-navigable picker. Dismisses with the chosen `SelectItem`, or
    `None` on cancel (escape) — agent-agnostic, reusable for any "pick one of these"
    interaction (currently `/model`)."""

    DEFAULT_CSS = """
    SelectScreen {
        align: center middle;
    }
    SelectScreen > OptionList {
        width: 60%;
        max-height: 70%;
        border: round $primary;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, items: list[SelectItem]) -> None:
        super().__init__()
        self._items = items

    def compose(self):
        options = [
            Option(
                item.label if not item.description else f"{item.label} — {item.description}",
                id=item.id,
            )
            for item in self._items
        ]
        yield OptionList(*options)

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        item = next((i for i in self._items if i.id == event.option.id), None)
        self.dismiss(item)

    def action_cancel(self) -> None:
        self.dismiss(None)
