import json
import random
from pathlib import Path
from typing import ClassVar

from attractors.themes.theme import Theme


class ThemeManager:
    _themes: ClassVar[dict[str, Theme]] = {}
    _default_theme: ClassVar[str] = "ayu"

    @classmethod
    def load(cls, path: str | Path) -> None:
        """Load themes from JSON file"""
        with open(path) as f:
            theme_data = json.load(f)

        for name, data in theme_data.items():
            cls._themes[name] = Theme(
                name=name,
                background=data["background"],
                colors=data["colors"],
                foreground=data.get("foreground", "#FFFFFF"),
            )

    @classmethod
    def get(cls, name: str | None = None) -> Theme:
        """Get theme by name or default"""
        name = name or cls._default_theme
        if name not in cls._themes:
            msg = f"Theme '{name}' not found"
            raise KeyError(msg)
        return cls._themes[name]

    @classmethod
    def list_themes(cls) -> list[str]:
        """List available themes"""
        return list(cls._themes.keys())

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set default theme"""
        if name not in cls._themes:
            msg = f"Theme '{name}' not found"
            raise KeyError(msg)
        cls._default_theme = name

    @classmethod
    def add(cls, theme: Theme) -> None:
        """Add new theme"""
        cls._themes[theme.name] = theme

    @classmethod
    def remove(cls, name: str) -> None:
        """Remove theme"""
        if name == cls._default_theme:
            raise ValueError("Cannot remove default theme")
        cls._themes.pop(name)

    @classmethod
    def random(cls) -> Theme:
        """Get random theme"""
        return random.choice(list(cls._themes.values()))
