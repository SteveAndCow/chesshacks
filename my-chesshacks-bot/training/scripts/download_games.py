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

    print(f"Downloading Lichess games for {month}...")
    print(f"This will download to: {output_dir}")
    print("\nNote: Files are compressed with zstd. You'll need to decompress them.")
    print("Install zstd: pip install zstandard")
    print("\nFor faster iteration, consider:")
    print("1. Download only high-rated games (2000+ ELO)")
    print("2. Use lichess.org/api to download specific game types")
    print("3. Start with 100k games, not millions")

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
    # For hackathon speed, recommend using Lichess API for targeted downloads
    print("="*60)
    print("QUICK START FOR HACKATHON")
    print("="*60)
    print("\nOption 1: Small curated dataset (RECOMMENDED for 36 hours)")
    print("  - Use Lichess API to get 10k-100k high-quality games")
    print("  - Filter by rating (2000+), time control (rapid/blitz)")
    print("  - Faster to download and process")
    print("\nOption 2: Pre-processed datasets")
    print("  - Search for 'chess dataset' on Kaggle/HuggingFace")
    print("  - Already in tensor format, ready to train")
    print("\nOption 3: Full database (time-consuming)")
    download_lichess_games()

    print("\n" + "="*60)
    print("Next step: Run preprocess.py to convert PGN to training data")
