"""
Download and filter chess games from Lichess database for training.

Efficiently downloads games from Lichess monthly database dumps and filters
by elo rating during decompression.

Usage:
    python download_games.py --min-elo 2000 --max-games 100000
"""
import argparse
import re
import requests
from pathlib import Path
from tqdm import tqdm
import sys

def download_and_filter_games(
    min_elo: int = 2000,
    max_games: int = 100000,
    month: str = "2024-10",
    output_dir: str = "training/data/raw",
    output_filename: str = "filtered_games.pgn"
):
    """
    Download and filter games from Lichess database.

    Args:
        min_elo: Minimum elo rating (both players must meet this)
        max_games: Maximum number of games to collect
        month: Which month to download (format: YYYY-MM)
        output_dir: Where to save the filtered PGN file
        output_filename: Name of output file
    """
    # Check if zstandard is available
    try:
        import zstandard as zstd
    except ImportError:
        print("❌ Error: zstandard library not found")
        print("Install it with: pip install zstandard")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lichess database URL
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    compressed_path = output_dir / f"lichess_{month}.pgn.zst"
    output_path = output_dir / output_filename

    print("="*60)
    print("LICHESS GAME DOWNLOAD & FILTER")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Minimum Elo: {min_elo} (both players)")
    print(f"  Target Games: {max_games:,}")
    print(f"  Source Month: {month}")
    print(f"  Output: {output_path}")

    # Download compressed file
    print(f"\n📥 Downloading compressed database...")
    print(f"URL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(compressed_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ Downloaded to: {compressed_path}")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Download manually with:")
        print(f"  wget {url} -O {compressed_path}")
        sys.exit(1)

    # Decompress and filter
    print(f"\n🔍 Filtering games (min elo: {min_elo})...")
    print("This may take a few minutes...")

    games_processed = 0
    games_kept = 0
    current_game = []
    white_elo = None
    black_elo = None
    keep_game = False

    with open(compressed_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed) as reader:
            with open(output_path, 'w', encoding='utf-8') as output:
                text_reader = reader.read(1024 * 1024)  # Read in 1MB chunks

                while text_reader:
                    lines = text_reader.decode('utf-8', errors='ignore').split('\n')

                    for line in lines:
                        # Parse elo ratings from headers
                        if line.startswith('[WhiteElo'):
                            match = re.search(r'"(\d+)"', line)
                            if match:
                                white_elo = int(match.group(1))

                        elif line.startswith('[BlackElo'):
                            match = re.search(r'"(\d+)"', line)
                            if match:
                                black_elo = int(match.group(1))

                        # Add line to current game
                        current_game.append(line)

                        # Empty line indicates end of game
                        if line.strip() == '' and current_game:
                            games_processed += 1

                            # Check if both players meet elo requirement
                            if white_elo and black_elo and white_elo >= min_elo and black_elo >= min_elo:
                                # Write game to output
                                output.write('\n'.join(current_game) + '\n')
                                games_kept += 1

                                # Progress update every 1000 games kept
                                if games_kept % 1000 == 0:
                                    filter_rate = (games_kept / games_processed) * 100
                                    print(f"  Kept: {games_kept:,} / Processed: {games_processed:,} ({filter_rate:.1f}%)")

                                # Stop if we've reached target
                                if games_kept >= max_games:
                                    print(f"\n✅ Reached target of {max_games:,} games!")
                                    break

                            # Reset for next game
                            current_game = []
                            white_elo = None
                            black_elo = None

                        # Break if we've collected enough games
                        if games_kept >= max_games:
                            break

                    # Break outer loop if target reached
                    if games_kept >= max_games:
                        break

                    # Read next chunk
                    text_reader = reader.read(1024 * 1024)

    # Cleanup compressed file
    print(f"\n🧹 Cleaning up compressed file...")
    compressed_path.unlink()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Games processed: {games_processed:,}")
    print(f"Games kept: {games_kept:,}")
    print(f"Filter rate: {(games_kept / games_processed) * 100:.1f}%")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    print("\n✅ Ready for preprocessing!")
    print(f"Next step: python preprocess.py --pgn {output_path} --output training/data/processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and filter chess games from Lichess database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 100k games with 2000+ elo
  python download_games.py --min-elo 2000 --max-games 100000

  # Download 50k games with 2200+ elo from January 2024
  python download_games.py --min-elo 2200 --max-games 50000 --month 2024-01
        """
    )

    parser.add_argument(
        '--min-elo',
        type=int,
        default=2000,
        help='Minimum elo rating (both players must meet this, default: 2000)'
    )

    parser.add_argument(
        '--max-games',
        type=int,
        default=100000,
        help='Maximum number of games to collect (default: 100000)'
    )

    parser.add_argument(
        '--month',
        type=str,
        default='2024-10',
        help='Lichess database month to download (format: YYYY-MM, default: 2024-10)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='training/data/raw',
        help='Output directory (default: training/data/raw)'
    )

    parser.add_argument(
        '--output-filename',
        type=str,
        default='filtered_games.pgn',
        help='Output filename (default: filtered_games.pgn)'
    )

    args = parser.parse_args()

    download_and_filter_games(
        min_elo=args.min_elo,
        max_games=args.max_games,
        month=args.month,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )
