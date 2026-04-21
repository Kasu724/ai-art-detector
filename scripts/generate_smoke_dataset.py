from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_art_detector.data.smoke import generate_smoke_dataset


if __name__ == "__main__":
    output_dir = Path("data/raw")
    generate_smoke_dataset(output_dir)
    print(f"Generated smoke dataset in {output_dir}")
