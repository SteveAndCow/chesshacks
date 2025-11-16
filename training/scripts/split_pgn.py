"""
Split a large PGN file into smaller chunks for parallel processing.

Usage:
    python training/scripts/split_pgn.py training/data/raw/filtered_games.pgn --games-per-chunk 5000
"""
import argparse
from pathlib import Path
import chess.pgn


def split_pgn(input_file: str, output_dir: str, games_per_chunk: int = 5000):
    """
    Split a large PGN file into smaller chunks.

    Args:
        input_file: Path to large PGN file
        output_dir: Directory to save chunks
        games_per_chunk: Number of games per output file
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Splitting {input_file}...")
    print(f"Target: {games_per_chunk:,} games per chunk")
    print(f"Output: {output_dir}")

    chunk_num = 0
    games_in_chunk = 0
    total_games = 0
    output_file = None

    with open(input_path) as pgn_file:
        while True:
            # Read game
            offset = pgn_file.tell()
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            # Start new chunk if needed
            if games_in_chunk == 0:
                if output_file:
                    output_file.close()
                    print(f"  Chunk {chunk_num-1:04d}: {games_per_chunk:,} games")

                chunk_filename = output_path / f"{input_path.stem}_chunk{chunk_num:04d}.pgn"
                output_file = open(chunk_filename, 'w')

            # Write game to current chunk
            # We need to re-read the raw text since chess.pgn.read_game() consumes it
            end_offset = pgn_file.tell()
            pgn_file.seek(offset)
            game_text = pgn_file.read(end_offset - offset)
            output_file.write(game_text)
            pgn_file.seek(end_offset)

            games_in_chunk += 1
            total_games += 1

            # Print progress
            if total_games % 10000 == 0:
                print(f"  Processed {total_games:,} games...")

            # Check if chunk is full
            if games_in_chunk >= games_per_chunk:
                games_in_chunk = 0
                chunk_num += 1

    # Close last chunk
    if output_file:
        output_file.close()
        print(f"  Chunk {chunk_num:04d}: {games_in_chunk:,} games")

    print(f"\n{'='*60}")
    print(f"Splitting complete!")
    print(f"{'='*60}")
    print(f"Total games: {total_games:,}")
    print(f"Chunks created: {chunk_num + 1}")
    print(f"Avg games per chunk: {total_games / (chunk_num + 1):,.0f}")

    # Estimate file sizes
    input_size_mb = input_path.stat().st_size / 1e6
    avg_chunk_size_mb = input_size_mb / (chunk_num + 1)
    print(f"\nInput size: {input_size_mb:,.0f} MB")
    print(f"Avg chunk size: {avg_chunk_size_mb:,.0f} MB")

    print(f"\n✅ Ready to upload!")
    print(f"Upload with:")
    print(f"  modal volume put chess-training-data {output_dir}/*.pgn /pgn/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large PGN file into chunks")
    parser.add_argument("input_file", help="Input PGN file")
    parser.add_argument("--output-dir", default="training/data/chunked",
                        help="Output directory for chunks")
    parser.add_argument("--games-per-chunk", type=int, default=5000,
                        help="Number of games per chunk")

    args = parser.parse_args()

    split_pgn(args.input_file, args.output_dir, args.games_per_chunk)
