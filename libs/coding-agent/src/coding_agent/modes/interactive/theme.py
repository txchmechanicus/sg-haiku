from __future__ import annotations

from textual.theme import Theme

# Ported from ~/code/stgzr/apps/aip's shared design system
# (starship/src/styles/starship.css `.dark { ... }` block, "inspired by Claude.ai").
# Only the values that theme actually defines are set explicitly here; anything it
# leaves unspecified (success/warning) uses a close Tailwind-palette match instead of
# Textual's defaults, to stay in the same visual family.
STARGAZER_DARK = Theme(
    name="stargazer-dark",
    dark=True,
    background="#1a1a1a",
    surface="#242424",
    panel="#2A2A2A",
    boost="#2E2E2E",
    primary="#c2a4f0",
    secondary="#7B5FC7",
    accent="#c2a4f0",
    foreground="#fafafa",
    error="#F87171",
    success="#4ADE80",
    warning="#FBBF24",
    variables={
        "border": "#3A3A3A",
        "text-muted": "#a3a3a3",
        "input-selection-background": "#643fb2 35%",
    },
)
