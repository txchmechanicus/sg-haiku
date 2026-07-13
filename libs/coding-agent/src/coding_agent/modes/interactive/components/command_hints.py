from __future__ import annotations

from textual.widgets import OptionList


class CommandHints(OptionList):
    """Filtered slash-command suggestion list shown above the prompt input while typing
    a command name. Purely visual/programmatically driven — the prompt `Input` keeps
    focus throughout; `HaikuApp.on_key` drives navigation/selection directly via
    `action_cursor_up`/`action_cursor_down` rather than this widget taking focus itself.
    """

    DEFAULT_CSS = """
    CommandHints {
        height: auto;
        max-height: 8;
        border: round $primary;
        display: none;
    }
    CommandHints.-visible {
        display: block;
    }
    """

    can_focus = False
