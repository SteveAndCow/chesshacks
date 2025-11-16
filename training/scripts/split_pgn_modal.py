"""
Split large PGN file on Modal (cloud-based, no local processing).

Upload the big file once, then split it on Modal workers.

Usage:
    # Upload big file
    modal volume put chess-training-data training/data/raw/filtered_games.pgn /raw/

    # Split it on Modal
    modal run training/scripts/split_pgn_modal.py --input-file filtered_games.pgn --games-per-chunk 5000
"""
import modal

app = modal.App("chesshacks-pgn-splitter")

image = modal.Image.debian_slim(python_version="3.11").pip_install("chess")

data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    cpu=4,
    timeout=7200,  # 2 hours max
)
def split_pgn_on_modal(input_file: str, games_per_chunk: int = 5000):
    """Split a large PGN file stored in Modal volume."""
    from pathlib import Path

    input_path = Path(f"/data/raw/{input_file}")
    output_dir = Path("/data/pgn")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return {"error": f"File not found: {input_path}"}

    print(f"Splitting {input_file}...")
    input_size_mb = input_path.stat().st_size / 1e6
    print(f"Input size: {input_size_mb:,.0f} MB")

    chunk_num = 0
    games_in_chunk = 0
    total_games = 0
    output_file = None
    game_buffer = []

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f):
            # Progress every 10M lines
            if line_num > 0 and line_num % 10_000_000 == 0:
                print(f"  Line {line_num:,}, Games {total_games:,}, Chunk {chunk_num}")

            # Detect game boundary
            if line.startswith('[Event '):
                # Save previous game
                if game_buffer:
                    # Start new chunk if needed
                    if games_in_chunk == 0:
                        if output_file:
                            output_file.close()
                            print(f"  ✓ Chunk {chunk_num-1:04d}: {games_per_chunk:,} games")

                        chunk_filename = output_dir / f"{input_path.stem}_chunk{chunk_num:04d}.pgn"
                        output_file = open(chunk_filename, 'w')

                    # Write game
                    output_file.write(''.join(game_buffer))
                    games_in_chunk += 1
                    total_games += 1

                    # Check if chunk full
                    if games_in_chunk >= games_per_chunk:
                        games_in_chunk = 0
                        chunk_num += 1

                    game_buffer = []

            game_buffer.append(line)

        # Write final game
        if game_buffer and output_file:
            output_file.write(''.join(game_buffer))
            games_in_chunk += 1
            total_games += 1

    if output_file:
        output_file.close()
        print(f"  ✓ Chunk {chunk_num:04d}: {games_in_chunk:,} games")

    # Commit volume
    data_volume.commit()

    print(f"\n{'='*60}")
    print(f"Splitting complete!")
    print(f"Total games: {total_games:,}")
    print(f"Chunks created: {chunk_num + 1}")
    print(f"{'='*60}")

    return {
        "total_games": total_games,
        "chunks_created": chunk_num + 1,
    }


@app.local_entrypoint()
def main(input_file: str = "filtered_games.pgn", games_per_chunk: int = 5000):
    """Split PGN file on Modal."""
    print(f"🚀 Splitting {input_file} on Modal...")
    print(f"Games per chunk: {games_per_chunk:,}")

    result = split_pgn_on_modal.remote(input_file, games_per_chunk)

    print(f"\n✅ Split complete!")
    print(f"Results: {result}")
    print(f"\nNext: Run preprocessing with:")
    print(f"  modal run training/scripts/preprocess_modal_lc0.py")
