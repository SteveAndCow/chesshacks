"""
FAST PGN splitter using text parsing (no chess library overhead).

This is 10-20x faster than chess.pgn parsing for large files.

Usage:
    python training/scripts/split_pgn_fast.py training/data/raw/filtered_games.pgn --games-per-chunk 5000
"""
import argparse
from pathlib import Path


def split_pgn_fast(input_file: str, output_dir: str, games_per_chunk: int = 5000):
    """
    Split PGN by detecting game boundaries (lines starting with '[Event').

    Args:
        input_file: Path to large PGN file
        output_dir: Directory to save chunks
        games_per_chunk: Number of games per output file
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Fast-splitting {input_file}...")
    print(f"Target: {games_per_chunk:,} games per chunk")
    print(f"Output: {output_dir}")

    chunk_num = 0
    games_in_chunk = 0
    total_games = 0
    current_chunk = []
    output_file = None

    input_size_mb = input_path.stat().st_size / 1e6
    print(f"Input size: {input_size_mb:,.0f} MB")

    with open(input_path, 'r') as f:
        game_buffer = []

        for line_num, line in enumerate(f):
            # Progress indicator (every 10M lines ≈ every 500MB for typical PGN)
            if line_num > 0 and line_num % 10_000_000 == 0:
                print(f"  Line {line_num:,}, Games {total_games:,}, Chunk {chunk_num}")

            # Detect start of new game
            if line.startswith('[Event '):
                # Save previous game if exists
                if game_buffer:
                    # Start new chunk if needed
                    if games_in_chunk == 0:
                        if output_file:
                            output_file.close()
                            print(f"  ✓ Chunk {chunk_num-1:04d}: {games_per_chunk:,} games")

                        chunk_filename = output_path / f"{input_path.stem}_chunk{chunk_num:04d}.pgn"
                        output_file = open(chunk_filename, 'w')

                    # Write game to current chunk
                    output_file.write(''.join(game_buffer))

                    games_in_chunk += 1
                    total_games += 1

                    # Check if chunk is full
                    if games_in_chunk >= games_per_chunk:
                        games_in_chunk = 0
                        chunk_num += 1

                    # Clear buffer
                    game_buffer = []

            # Add line to current game
            game_buffer.append(line)

        # Write final game
        if game_buffer and output_file:
            output_file.write(''.join(game_buffer))
            games_in_chunk += 1
            total_games += 1

    # Close last chunk
    if output_file:
        output_file.close()
        print(f"  ✓ Chunk {chunk_num:04d}: {games_in_chunk:,} games")

    print(f"\n{'='*60}")
    print(f"Splitting complete!")
    print(f"{'='*60}")
    print(f"Total games: {total_games:,}")
    print(f"Chunks created: {chunk_num + 1}")
    print(f"Avg games per chunk: {total_games / (chunk_num + 1):,.0f}")

    # Estimate chunk sizes
    avg_chunk_size_mb = input_size_mb / (chunk_num + 1)
    print(f"Avg chunk size: {avg_chunk_size_mb:,.0f} MB")

    print(f"\n✅ Ready to upload!")
    print(f"Upload with:")
    print(f"  modal volume put chess-training-data {output_dir}/*.pgn /pgn/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast PGN splitter")
    parser.add_argument("input_file", help="Input PGN file")
    parser.add_argument("--output-dir", default="training/data/chunked",
                        help="Output directory for chunks")
    parser.add_argument("--games-per-chunk", type=int, default=5000,
                        help="Number of games per chunk (default: 5000)")

    args = parser.parse_args()

    split_pgn_fast(args.input_file, args.output_dir, args.games_per_chunk)
