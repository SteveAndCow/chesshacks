"""
Download chess games from Lichess database for training.
For a 36-hour hackathon, start with a smaller dataset (~100k-1M games).
"""
import requests
import os
from pathlib import Path

def download_lichess_games(output_dir="training/data/raw", month="2024-01", max_games=100000):
    """
    Download games from Lichess database.

    Args:
        output_dir: Where to save the downloaded PGN files
        month: Which month to download (format: YYYY-MM)
        max_games: Approximate number of games to download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lichess database URLs
    # For speed, download from specific rating brackets
    urls = [
        f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst",
    ]

    for url in urls:
        filename = url.split("/")[-1]
        output_path = output_dir / filename

        print(f"\nDownloading: {filename}")
        print(f"URL: {url}")
        print(f"Save to: {output_path}")
        print("\nRun this command to download:")
        print(f"wget {url} -O {output_path}")
        print(f"# Or: curl -L {url} -o {output_path}")

if __name__ == "__main__":
    download_lichess_games()

    print("\n" + "="*60)
    print("Next step: Run preprocess.py to convert PGN to training data")
