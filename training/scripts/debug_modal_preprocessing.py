"""
Debug script to test Modal preprocessing.
Simplified version that will show exactly what's happening.
"""
import modal
from pathlib import Path

app = modal.App("chesshacks-debug")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("chess", "numpy")
)

data_volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)


@app.function(
    image=image,
    cpu=2,
    timeout=600,
    volumes={"/data": data_volume},
)
def debug_pgn_file():
    """Debug the PGN file to see what's wrong."""
    import chess.pgn
    from pathlib import Path

    print("="*60)
    print("DEBUG: Checking PGN file")
    print("="*60)

    # Check if pgn directory exists
    pgn_dir = Path("/data/pgn")
    print(f"\n1. Checking if {pgn_dir} exists...")
    if not pgn_dir.exists():
        print(f"   ❌ Directory does not exist!")
        return
    print(f"   ✅ Directory exists")

    # List files
    print(f"\n2. Listing files in {pgn_dir}...")
    files = list(pgn_dir.iterdir())
    if not files:
        print(f"   ❌ No files found!")
        return
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"   - {f.name}: {size_mb:.2f} MB")

    # Try to open first file
    pgn_file = files[0]
    print(f"\n3. Opening {pgn_file.name}...")

    try:
        with open(pgn_file) as f:
            # Read first 10 games
            for i in range(10):
                game = chess.pgn.read_game(f)
                if game is None:
                    print(f"   Reached end of file at game {i}")
                    break

                # Get game info
                white = game.headers.get("White", "Unknown")
                black = game.headers.get("Black", "Unknown")
                result = game.headers.get("Result", "*")
                white_elo = game.headers.get("WhiteElo", "?")
                black_elo = game.headers.get("BlackElo", "?")

                print(f"\n   Game {i+1}:")
                print(f"     White: {white} ({white_elo})")
                print(f"     Black: {black} ({black_elo})")
                print(f"     Result: {result}")
                print(f"     Moves: {len(list(game.mainline_moves()))}")

        print(f"\n✅ File is readable and contains games!")

        # Now test with ELO filter
        print(f"\n4. Testing with ELO >= 2000 filter...")
        with open(pgn_file) as f:
            count = 0
            filtered = 0
            for i in range(100):  # Check first 100 games
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))

                    if white_elo >= 2000 and black_elo >= 2000:
                        count += 1
                    else:
                        filtered += 1
                except ValueError:
                    filtered += 1

            print(f"   Checked first 100 games:")
            print(f"   - Passed filter (>= 2000 ELO): {count}")
            print(f"   - Filtered out: {filtered}")

        if count == 0:
            print(f"\n   ⚠️  WARNING: No games found with ELO >= 2000 in first 100!")
            print(f"   Try lowering --min-elo to 1500 or 1800")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    """Run debug check."""
    print("Running debug check on Modal...")
    debug_pgn_file.remote()
    print("\n✅ Debug complete")
