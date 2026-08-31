from pathlib import Path


path = Path("RESULT.txt")
raise SystemExit(0 if path.is_file() and path.read_text(encoding="utf-8") == "ok" else 1)
