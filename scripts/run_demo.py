import streamlit.web.cli as stcli
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_art_detector.utils.env import load_project_env


if __name__ == "__main__":
    load_project_env()
    sys.argv = ["streamlit", "run", "src/ai_art_detector/demo/app.py"]
    raise SystemExit(stcli.main())
