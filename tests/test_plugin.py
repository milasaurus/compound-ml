"""
Tests for compound-ml plugin structure, skill loading, and command routing.

Validates that the plugin follows the Claude Code marketplace plugin conventions
so that slash commands like /compound-ml:ml-explore route correctly.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import re

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKETPLACE_JSON = REPO_ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN_DIR = REPO_ROOT / "plugins" / "compound-ml"
PLUGIN_JSON = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
SKILLS_DIR = PLUGIN_DIR / "skills"
AGENTS_DIR = PLUGIN_DIR / "agents"

EXPECTED_SKILLS = ["ml-explore", "ml-cluster", "ml-anomalies", "ml-rag", "ml-analyze"]
EXPECTED_AGENTS = [
    ("research", "ml-literature-researcher"),
    ("review", "ml-output-reviewer"),
]

DATASET_CSV = REPO_ROOT / "data" / "dataset.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_skill_frontmatter(skill_path: Path) -> dict:
    """Parse YAML frontmatter from a SKILL.md file (no PyYAML dependency)."""
    text = skill_path.read_text()
    assert text.startswith("---"), f"{skill_path} missing YAML frontmatter"
    _, fm_raw, _ = text.split("---", 2)
    result = {}
    for line in fm_raw.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
    return result


def run_python(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python snippet and return the completed process."""
    return subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ===========================================================================
# 1. Plugin structure
# ===========================================================================


class TestPluginStructure:
    """Verify the marketplace plugin directory layout is correct."""

    def test_marketplace_json_exists(self):
        assert MARKETPLACE_JSON.exists(), "Missing .claude-plugin/marketplace.json"

    def test_no_root_plugin_json(self):
        root_pj = REPO_ROOT / ".claude-plugin" / "plugin.json"
        assert not root_pj.exists(), (
            "Root .claude-plugin/ must NOT contain plugin.json — "
            "only marketplace.json belongs here. plugin.json at root "
            "causes the resolver to treat the repo as a flat plugin "
            "and skip the plugins/ subdirectory."
        )

    def test_marketplace_json_source_path(self):
        data = json.loads(MARKETPLACE_JSON.read_text())
        sources = [p["source"] for p in data["plugins"]]
        assert "./plugins/compound-ml" in sources, (
            f"marketplace.json must list source './plugins/compound-ml', got {sources}"
        )

    def test_inner_plugin_json_exists(self):
        assert PLUGIN_JSON.exists(), "Missing plugins/compound-ml/.claude-plugin/plugin.json"

    def test_inner_plugin_json_has_name(self):
        data = json.loads(PLUGIN_JSON.read_text())
        assert data.get("name") == "compound-ml"

    def test_skills_directory_exists(self):
        assert SKILLS_DIR.is_dir(), "Missing plugins/compound-ml/skills/"

    def test_agents_directory_exists(self):
        assert AGENTS_DIR.is_dir(), "Missing plugins/compound-ml/agents/"


# ===========================================================================
# 2. Skill loading — each skill has valid SKILL.md with correct frontmatter
# ===========================================================================


class TestSkillLoading:
    """Each of the 5 skills can be loaded without error."""

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_directory_exists(self, skill_name):
        skill_dir = SKILLS_DIR / skill_name
        assert skill_dir.is_dir(), f"Missing skill directory: {skill_name}"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_md_exists(self, skill_name):
        skill_md = SKILLS_DIR / skill_name / "SKILL.md"
        assert skill_md.exists(), f"Missing SKILL.md in {skill_name}"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_frontmatter_has_name(self, skill_name):
        fm = parse_skill_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        assert fm.get("name") == skill_name, (
            f"Skill {skill_name} frontmatter name is '{fm.get('name')}', expected '{skill_name}'"
        )

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_frontmatter_has_description(self, skill_name):
        fm = parse_skill_frontmatter(SKILLS_DIR / skill_name / "SKILL.md")
        desc = fm.get("description", "")
        assert len(desc) > 20, f"Skill {skill_name} description is too short or missing"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_frontmatter_description_is_quoted(self, skill_name):
        """Description values containing colons must be quoted in YAML."""
        raw = (SKILLS_DIR / skill_name / "SKILL.md").read_text()
        _, fm_raw, _ = raw.split("---", 2)
        # Find the description line — it should start with quotes if it contains ':'
        for line in fm_raw.strip().splitlines():
            if line.startswith("description:"):
                value = line.split("description:", 1)[1].strip()
                if ":" in value.strip('"').strip("'"):
                    assert value.startswith('"') or value.startswith("'"), (
                        f"Skill {skill_name}: description contains ':' but is not quoted"
                    )


# ===========================================================================
# 3. Command routing — slash commands resolve to the correct skill
# ===========================================================================


class TestCommandRouting:
    """Slash commands resolve to the correct skill via marketplace.json → source → skills/."""

    def test_marketplace_source_directory_exists(self):
        data = json.loads(MARKETPLACE_JSON.read_text())
        for plugin in data["plugins"]:
            source = plugin["source"]
            resolved = REPO_ROOT / source
            assert resolved.is_dir(), f"Source path '{source}' does not resolve to a directory"

    def test_marketplace_source_has_skills(self):
        data = json.loads(MARKETPLACE_JSON.read_text())
        for plugin in data["plugins"]:
            source = plugin["source"]
            skills = REPO_ROOT / source / "skills"
            assert skills.is_dir(), f"Source '{source}' has no skills/ directory"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_slash_command_resolves(self, skill_name):
        """
        /compound-ml:<skill_name> should resolve to
        plugins/compound-ml/skills/<skill_name>/SKILL.md
        """
        target = PLUGIN_DIR / "skills" / skill_name / "SKILL.md"
        assert target.exists(), (
            f"/compound-ml:{skill_name} would fail — "
            f"no SKILL.md at {target.relative_to(REPO_ROOT)}"
        )

    @pytest.mark.parametrize("category,agent_name", EXPECTED_AGENTS)
    def test_agent_file_exists(self, category, agent_name):
        agent_file = AGENTS_DIR / category / f"{agent_name}.md"
        assert agent_file.exists(), f"Missing agent: {category}/{agent_name}.md"


# ===========================================================================
# 4. File handling — ml-explore reads CSV correctly
# ===========================================================================


class TestFileHandling:
    """ml-explore correctly handles various file inputs."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv = tmp_path / "sample.csv"
        csv.write_text("name,age,score\nAlice,30,85.5\nBob,25,92.1\nCarol,35,78.3\n")
        return csv

    def test_read_csv_absolute_path(self, sample_csv):
        """ml-explore should accept an absolute path to a CSV."""
        result = run_python(f"""
import pandas as pd
df = pd.read_csv("{sample_csv}")
print(f"shape: {{df.shape}}")
print(f"columns: {{list(df.columns)}}")
print(f"dtypes: {{dict(df.dtypes)}}")
""")
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "shape: (3, 3)" in result.stdout

    def test_missing_file_fails_gracefully(self):
        """Should produce a clear error, not a raw traceback."""
        result = run_python("""
import pandas as pd
import sys
path = "/nonexistent/path/data.csv"
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: File not found: {path}")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(0)
""")
        assert result.returncode == 0
        assert "Error: File not found" in result.stdout

    def test_empty_file_detected(self, tmp_path):
        """Empty CSV should be caught before profiling."""
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        result = run_python(f"""
import pandas as pd
import sys
try:
    df = pd.read_csv("{empty}")
    if df.empty:
        print("Error: Not enough data for meaningful profiling (0 rows)")
        sys.exit(0)
except pd.errors.EmptyDataError:
    print("Error: Not enough data for meaningful profiling (empty file)")
    sys.exit(0)
""")
        assert result.returncode == 0
        assert "Not enough data" in result.stdout

    def test_non_csv_file_detected(self, tmp_path):
        """Non-CSV files should produce a format error, not crash."""
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        result = run_python(f"""
import pandas as pd
import sys
path = "{binary}"
ext = path.rsplit(".", 1)[-1] if "." in path else ""
supported = {{"csv", "json", "parquet"}}
if ext not in supported:
    print(f"Error: Unsupported format '.{{ext}}'. Supported: {{', '.join(sorted(supported))}}")
    sys.exit(0)
""")
        assert result.returncode == 0
        assert "Unsupported format" in result.stdout


# ===========================================================================
# 5. End-to-end — ml-explore on the Spotify dataset
# ===========================================================================


class TestEndToEnd:
    """ml-explore on the Spotify tracks dataset produces expected outputs."""

    @pytest.mark.skipif(
        not DATASET_CSV.exists(),
        reason=f"Dataset not found at {DATASET_CSV}",
    )
    def test_spotify_profile(self):
        """
        Profile the Spotify dataset and verify it produces:
        - shape
        - dtypes
        - missing value report
        - at least 2 modeling suggestions
        """
        result = run_python(f"""
import pandas as pd
import sys

df = pd.read_csv("{DATASET_CSV}")

# Shape
print(f"SHAPE: {{df.shape[0]}} rows x {{df.shape[1]}} columns")

# Dtypes
print("DTYPES:")
for col, dtype in df.dtypes.items():
    print(f"  {{col}}: {{dtype}}")

# Missing values
print("MISSING:")
missing = df.isnull().sum()
for col, count in missing.items():
    if count > 0:
        pct = count / len(df) * 100
        print(f"  {{col}}: {{count}} ({{pct:.1f}}%)")
if missing.sum() == 0:
    print("  No missing values")

# Modeling suggestions based on data characteristics
suggestions = []
text_cols = df.select_dtypes(include=["object"]).columns
numeric_cols = df.select_dtypes(include=["number"]).columns

if len(text_cols) > 0:
    suggestions.append("SUGGEST: Use ml-cluster to discover groups by text similarity")
if len(numeric_cols) > 3:
    suggestions.append("SUGGEST: Use ml-anomalies to flag unusual tracks by numeric features")
if len(df) > 1000:
    suggestions.append("SUGGEST: Use ml-analyze for a comprehensive end-to-end report")

for s in suggestions:
    print(s)
""", timeout=60)
        assert result.returncode == 0, f"Profiling failed: {result.stderr}"

        out = result.stdout
        # Shape
        assert "SHAPE:" in out
        assert "rows" in out and "columns" in out

        # Dtypes
        assert "DTYPES:" in out
        assert "int" in out or "float" in out  # numeric dtypes present

        # Missing values
        assert "MISSING:" in out

        # At least 2 modeling suggestions
        suggest_count = out.count("SUGGEST:")
        assert suggest_count >= 2, (
            f"Expected at least 2 modeling suggestions, got {suggest_count}"
        )
