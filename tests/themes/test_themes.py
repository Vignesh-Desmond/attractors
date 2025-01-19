import json
from pathlib import Path

import matplotlib.colors
import pytest

from attractors.themes.manager import ThemeManager
from attractors.themes.theme import Theme


@pytest.fixture()
def sample_theme():
    """Create a sample theme for testing"""
    return Theme(
        name="test_theme",
        background="#000000",
        foreground="#FFFFFF",
        colors=["#FF0000", "#00FF00", "#0000FF"],
    )


@pytest.fixture()
def sample_theme_data():
    """Sample theme data for testing file loading"""
    return {
        "test_theme1": {
            "background": "#000000",
            "foreground": "#FFFFFF",
            "colors": ["#FF0000", "#00FF00"],
        },
        "test_theme2": {"background": "#FFFFFF", "colors": "viridis"},
    }


@pytest.fixture()
def theme_file(tmp_path, sample_theme_data):
    """Create a temporary theme file"""
    file_path = tmp_path / "test_themes.json"
    with open(file_path, "w") as f:
        json.dump(sample_theme_data, f)
    return file_path


class TestTheme:
    def test_theme_creation(self, sample_theme):
        """Test basic theme creation"""
        assert sample_theme.name == "test_theme"
        assert sample_theme.background == "#000000"
        assert sample_theme.foreground == "#FFFFFF"
        assert isinstance(sample_theme.colors, list)

    def test_colormap_property_list(self, sample_theme):
        """Test colormap creation from list of colors"""
        cmap = sample_theme.colormap
        assert isinstance(cmap, matplotlib.colors.LinearSegmentedColormap)

    def test_colormap_property_string(self):
        """Test colormap creation from string"""
        theme = Theme(
            name="test_theme", background="#000000", foreground="#FFFFFF", colors="viridis"
        )
        cmap = theme.colormap
        assert isinstance(cmap, matplotlib.colors.Colormap)


class TestThemeManager:
    def setup_method(self):
        """Save original default theme before each test"""
        self.original_default = ThemeManager._default_theme

    def teardown_method(self):
        """Restore original default theme after each test"""
        ThemeManager._default_theme = self.original_default
        theme_names = list(ThemeManager._themes.keys())
        for name in theme_names:
            if name != self.original_default:
                ThemeManager._themes.pop(name, None)

    def test_load_themes(self, theme_file):
        """Test loading themes from file"""
        ThemeManager.load(theme_file)
        assert "test_theme1" in ThemeManager._themes
        assert "test_theme2" in ThemeManager._themes

    def test_get_theme(self, sample_theme):
        """Test getting theme by name"""
        ThemeManager.add(sample_theme)
        theme = ThemeManager.get(sample_theme.name)
        assert theme.name == sample_theme.name
        assert theme.background == sample_theme.background

    def test_get_default_theme(self):
        """Test getting default theme"""
        theme = ThemeManager.get()
        assert isinstance(theme, Theme)
        assert theme.name == ThemeManager._default_theme

    def test_invalid_theme_access(self):
        """Test error handling for invalid theme access"""
        with pytest.raises(KeyError, match="Theme 'nonexistent_theme' not found"):
            ThemeManager.get("nonexistent_theme")

    def test_set_default_theme(self, sample_theme):
        """Test setting default theme"""
        ThemeManager.add(sample_theme)
        ThemeManager.set_default(sample_theme.name)
        assert ThemeManager._default_theme == sample_theme.name

    def test_invalid_default_theme(self):
        """Test setting invalid default theme"""
        with pytest.raises(KeyError):
            ThemeManager.set_default("nonexistent_theme")

    def test_list_themes(self, sample_theme):
        """Test listing available themes"""
        ThemeManager.add(sample_theme)
        themes = ThemeManager.list_themes()
        assert isinstance(themes, list)
        assert sample_theme.name in themes

    def test_add_remove_theme(self):
        """Test adding and removing themes"""
        default_name = ThemeManager._default_theme
        test_theme = Theme(
            name="non_default_test_theme",
            background="#000000",
            foreground="#FFFFFF",
            colors=["#FF0000", "#00FF00", "#0000FF"],
        )

        ThemeManager.add(test_theme)
        assert test_theme.name in ThemeManager._themes

        ThemeManager.remove(test_theme.name)
        assert test_theme.name not in ThemeManager._themes
        assert default_name == ThemeManager._default_theme  # default should remain unchanged

    def test_remove_default_theme(self):
        """Test attempting to remove default theme"""
        default_name = ThemeManager._default_theme
        with pytest.raises(ValueError, match="Cannot remove default theme"):
            ThemeManager.remove(default_name)

    def test_random_theme(self, sample_theme):
        """Test getting random theme"""
        ThemeManager.add(sample_theme)
        random_theme = ThemeManager.random()
        assert isinstance(random_theme, Theme)


def test_actual_theme_file():
    """Test that the actual viz_themes.json file is valid and loadable"""
    # assumes the test is run from the project root
    theme_path = Path("attractors/themes/viz_themes.json")
    if theme_path.exists():
        with open(theme_path) as f:
            themes_data = json.load(f)

        for name, data in themes_data.items():
            assert isinstance(name, str)
            assert "background" in data
            assert isinstance(data["background"], str)
            assert "colors" in data
            assert isinstance(data["colors"], str | list)

        ThemeManager.load(theme_path)
        assert len(ThemeManager._themes) > 0
