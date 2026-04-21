import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn


if __name__ == "__main__":
    uvicorn.run("ai_art_detector.api.app:app", host="0.0.0.0", port=8000, reload=False)
