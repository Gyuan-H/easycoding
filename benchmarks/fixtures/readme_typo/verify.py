from pathlib import Path

text = Path("README.md").read_text(encoding="utf-8")
assert "teh project" not in text
assert "the project" in text

