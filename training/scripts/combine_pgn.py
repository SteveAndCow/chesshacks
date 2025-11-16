"""
Combine multiple PGN files into a single file.

Useful for:
- Combining TWIC weekly files into one dataset
- Merging different data sources (Lichess + TWIC)
- Preparing data for preprocessing

Usage:
    # Combine all TWIC files
    python combine_pgn.py --input-dir training/data/raw/twic_chess_games_2025 --output twic_2025_combined.pgn

    # Combine multiple specific files
    python combine_pgn.py --input-files file1.pgn file2.pgn --output combined.pgn

    # Combine TWIC with existing filtered_games.pgn
    python combine_pgn.py --input-files training/data/raw/filtered_games.pgn training/data/raw/twic_2025_combined.pgn --output all_games_combined.pgn
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import chess.pgn
import io
import warnings


def combine_pgn_files(input_files: list[Path], output_path: Path, deduplicate: bool = False):
    """
    Combine multiple PGN files into one using python-chess for reliable parsing.

    Args:
        input_files: List of PGN file paths to combine
        output_path: Output file path
        deduplicate: If True, skip duplicate games (by headers)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("COMBINING PGN FILES")
    print("="*60)
    print(f"\nInput files: {len(input_files)}")
    for f in input_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  - {f.name}: {size_mb:.1f} MB")
    print(f"\nOutput: {output_path}")

    total_games = 0
    total_size = 0
    skipped_games = 0
    seen_games = set() if deduplicate else None

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for pgn_file in tqdm(input_files, desc="Processing files"):
            if not pgn_file.exists():
                print(f"⚠️  Warning: {pgn_file} not found, skipping")
                continue

            file_size = pgn_file.stat().st_size
            total_size += file_size

            games_in_file = 0
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as infile:
                # Suppress python-chess warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    original_stderr = sys.stderr
                    sys.stderr = io.StringIO()

                    try:
                        while True:
                            try:
                                game = chess.pgn.read_game(infile)
                            except Exception:
                                # Restore stderr for error reporting
                                sys.stderr = original_stderr
                                raise
                            
                            if game is None:
                                break
                            
                            # Restore stderr after reading game
                            sys.stderr = original_stderr

                            # Deduplication check
                            if deduplicate:
                                # Create hash from key headers
                                key_headers = (
                                    game.headers.get("Event", ""),
                                    game.headers.get("Site", ""),
                                    game.headers.get("Date", ""),
                                    game.headers.get("Round", ""),
                                    game.headers.get("White", ""),
                                    game.headers.get("Black", ""),
                                )
                                game_hash = hash(key_headers)
                                if game_hash in seen_games:
                                    skipped_games += 1
                                    continue
                                seen_games.add(game_hash)

                            # Write game to output
                            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                            game_str = game.accept(exporter)
                            outfile.write(game_str)
                            outfile.write("\n\n")  # Extra newline between games

                            games_in_file += 1
                            total_games += 1

                            # Progress update
                            if total_games % 1000 == 0:
                                print(f"  Processed {total_games:,} games...")

                    except Exception as e:
                        sys.stderr = original_stderr
                        print(f"⚠️  Error processing {pgn_file.name}: {e}")
                        continue
                    finally:
                        # Always restore stderr
                        sys.stderr = original_stderr

            print(f"  {pgn_file.name}: {games_in_file:,} games")

    output_size_mb = output_path.stat().st_size / (1024**2)
    
    print("\n" + "="*60)
    print("COMBINE COMPLETE")
    print("="*60)
    print(f"Total games: {total_games:,}")
    if deduplicate:
        print(f"Duplicates skipped: {skipped_games:,}")
    print(f"Input size: {total_size / (1024**2):.1f} MB")
    print(f"Output size: {output_size_mb:.1f} MB")
    print(f"Output file: {output_path}")
    print("\n✅ Ready for preprocessing!")
    print(f"Next: python preprocess.py --input {output_path} --output training/data/processed")


def combine_directory(input_dir: Path, output_path: Path, pattern: str = "*.pgn", deduplicate: bool = False):
    """
    Combine all PGN files in a directory.

    Args:
        input_dir: Directory containing PGN files
        output_path: Output file path
        pattern: Glob pattern to match files (default: "*.pgn")
        deduplicate: If True, skip duplicate games
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"❌ Error: Directory {input_dir} does not exist")
        sys.exit(1)

    pgn_files = sorted(input_dir.glob(pattern))
    
    if not pgn_files:
        print(f"❌ Error: No PGN files found in {input_dir} matching pattern '{pattern}'")
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN files in {input_dir}")
    combine_pgn_files(pgn_files, output_path, deduplicate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple PGN files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all TWIC files from a directory
  python combine_pgn.py --input-dir training/data/raw/twic_chess_games_2025 --output twic_2025_combined.pgn

  # Combine specific files
  python combine_pgn.py --input-files file1.pgn file2.pgn --output combined.pgn

  # Combine TWIC with Lichess data
  python combine_pgn.py --input-files training/data/raw/filtered_games.pgn training/data/raw/twic_2025_combined.pgn --output all_games.pgn

  # Combine with deduplication (removes duplicate games)
  python combine_pgn.py --input-dir training/data/raw/twic_chess_games_2025 --output twic_2025_combined.pgn --deduplicate
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--input-dir',
        type=str,
        help='Directory containing PGN files to combine'
    )
    group.add_argument(
        '--input-files',
        nargs='+',
        help='List of PGN files to combine'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output PGN file path'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pgn',
        help='Glob pattern for files in directory (default: *.pgn)'
    )

    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Remove duplicate games (based on headers)'
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if args.input_dir:
        combine_directory(
            Path(args.input_dir),
            output_path,
            pattern=args.pattern,
            deduplicate=args.deduplicate
        )
    else:
        input_files = [Path(f) for f in args.input_files]
        combine_pgn_files(
            input_files,
            output_path,
            deduplicate=args.deduplicate
        )

