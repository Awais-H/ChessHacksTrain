"""
Process Lichess puzzle database for training.
Puzzles are PERFECT for winning intent - every puzzle has a correct winning move!

Lichess Puzzle DB: https://database.lichess.org/#puzzles
Format: CSV with FEN, Moves, Rating, Themes, etc.
"""

import os
import subprocess
import numpy as np
import chess
import csv
from tqdm import tqdm
from typing import Tuple, Optional, List

# Import from V2 for board encoding
from data_processing_v2 import board_to_array_v2, move_to_index, calculate_position_value


def download_puzzle_database(output_dir: str = ".") -> Optional[str]:
    """
    Download Lichess puzzle database.
    
    URL: https://database.lichess.org/lichess_db_puzzle.csv.zst
    
    Returns:
        Path to extracted CSV file
    """
    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    filename = "lichess_db_puzzle.csv.zst"
    zst_path = os.path.join(output_dir, filename)
    csv_path = zst_path.replace(".zst", "")
    
    # Skip if already extracted
    if os.path.exists(csv_path):
        print(f"✓ Puzzle CSV already exists: {csv_path}")
        return csv_path
    
    try:
        # Download
        if not os.path.exists(zst_path):
            print(f"Downloading Lichess puzzle database...")
            print(f"  URL: {url}")
            subprocess.run(["wget", "-q", url, "-O", zst_path], check=True)
            print(f"✓ Downloaded {filename}")
        
        # Extract
        print(f"Extracting {filename}...")
        subprocess.run(["unzstd", zst_path], check=True)
        print(f"✓ Extracted to {csv_path}")
        
        # Clean up compressed file
        if os.path.exists(zst_path):
            os.remove(zst_path)
            print(f"✓ Cleaned up {filename}")
        
        return csv_path
        
    except Exception as e:
        print(f"✗ Failed to download/extract: {e}")
        return None


def process_puzzle_database(
    csv_path: str,
    max_positions: int = 500000,
    min_rating: int = 1500,
    max_rating: int = 2500,
    include_values: bool = True,
    include_position_eval: bool = False,
    themes_filter: Optional[List[str]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process Lichess puzzle database.
    
    Puzzle CSV format:
    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
    
    Args:
        csv_path: Path to puzzle CSV
        max_positions: Maximum puzzles to extract
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
        include_values: Include piece value channel
        include_position_eval: Include position evaluation
        themes_filter: Optional list of themes to filter (e.g., ["mate", "endgame"])
        
    Returns:
        Tuple of (positions, moves, values)
    """
    if not os.path.exists(csv_path):
        print(f"✗ Puzzle CSV not found: {csv_path}")
        return None, None, None
    
    positions_list = []
    moves_list = []
    values_list = [] if include_position_eval else None
    
    print(f"\nProcessing Lichess Puzzle Database:")
    print(f"  File: {os.path.basename(csv_path)}")
    print(f"  Max positions: {max_positions}")
    print(f"  Rating range: {min_rating}-{max_rating}")
    if themes_filter:
        print(f"  Themes filter: {themes_filter}")
    print()
    
    puzzles_processed = 0
    puzzles_skipped = 0
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            pbar = tqdm(total=max_positions, desc="Processing puzzles", unit="puzzle")
            
            for row in reader:
                if len(positions_list) >= max_positions:
                    break
                
                try:
                    # Parse puzzle data
                    puzzle_id = row['PuzzleId']
                    fen = row['FEN']
                    moves_uci = row['Moves'].split()  # Space-separated UCI moves
                    rating = int(row['Rating'])
                    themes = row['Themes'].split()
                    
                    # Filter by rating
                    if rating < min_rating or rating > max_rating:
                        puzzles_skipped += 1
                        continue
                    
                    # Filter by themes (if specified)
                    if themes_filter:
                        if not any(theme in themes for theme in themes_filter):
                            puzzles_skipped += 1
                            continue
                    
                    # Setup board from FEN
                    board = chess.Board(fen)
                    
                    # First move is opponent's move (puzzle setup)
                    # Second move is the correct solution (what we want to learn!)
                    if len(moves_uci) < 2:
                        puzzles_skipped += 1
                        continue
                    
                    # Apply opponent's move
                    opponent_move = chess.Move.from_uci(moves_uci[0])
                    board.push(opponent_move)
                    
                    # Now board is in position where we need to find the solution
                    solution_move = chess.Move.from_uci(moves_uci[1])
                    
                    # Encode position and solution
                    position = board_to_array_v2(board, include_values)
                    move_idx = move_to_index(solution_move)
                    
                    positions_list.append(position)
                    moves_list.append(move_idx)
                    
                    if include_position_eval:
                        # Puzzles are winning positions by definition
                        # Calculate material advantage
                        value = calculate_position_value(board)
                        values_list.append(value)
                    
                    puzzles_processed += 1
                    pbar.update(1)
                    
                    if puzzles_processed % 1000 == 0:
                        pbar.set_postfix({
                            'processed': puzzles_processed,
                            'skipped': puzzles_skipped
                        })
                
                except Exception as e:
                    puzzles_skipped += 1
                    continue
            
            pbar.close()
    
    except Exception as e:
        print(f"✗ Error processing puzzles: {e}")
        return None, None, None
    
    print(f"\n✓ Processed {puzzles_processed} puzzles")
    print(f"  Skipped: {puzzles_skipped} puzzles")
    
    if not positions_list:
        print("✗ No puzzles extracted")
        return None, None, None
    
    positions = np.array(positions_list, dtype=np.float32)
    moves = np.array(moves_list, dtype=np.int32)
    values = np.array(values_list, dtype=np.float32) if include_position_eval else None
    
    print(f"✓ Final shapes: positions={positions.shape}, moves={moves.shape}")
    if values is not None:
        print(f"  values={values.shape}")
    
    return positions, moves, values


def load_or_process_puzzle_database(
    cache_dir: str,
    max_positions: int = 500000,
    min_rating: int = 1500,
    max_rating: int = 2500,
    include_values: bool = True,
    include_position_eval: bool = False,
    themes_filter: Optional[List[str]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cached puzzle dataset or download and process.
    
    Args:
        cache_dir: Directory for caching
        max_positions: Maximum puzzles to extract
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
        include_values: Include piece value channel
        include_position_eval: Include position evaluation
        themes_filter: Optional list of themes (e.g., ["mate", "endgame"])
        
    Returns:
        Tuple of (positions, moves, values)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filenames
    suffix = f"_puzzles_r{min_rating}-{max_rating}"
    if themes_filter:
        suffix += f"_{'_'.join(themes_filter)}"
    if include_values:
        suffix += "_vals"
    if include_position_eval:
        suffix += "_eval"
    
    positions_cache = os.path.join(cache_dir, f"lichess_positions{suffix}.npy")
    moves_cache = os.path.join(cache_dir, f"lichess_moves{suffix}.npy")
    values_cache = os.path.join(cache_dir, f"lichess_values{suffix}.npy")
    
    # Check cache
    cache_exists = os.path.exists(positions_cache) and os.path.exists(moves_cache)
    if include_position_eval:
        cache_exists = cache_exists and os.path.exists(values_cache)
    
    if cache_exists:
        print(f"✓ Loading cached puzzle dataset")
        positions = np.load(positions_cache)
        moves = np.load(moves_cache)
        values = np.load(values_cache) if include_position_eval else None
        print(f"  Loaded {len(positions)} puzzles")
        return positions, moves, values
    
    # Download and process
    print(f"Processing Lichess puzzle database...")
    csv_path = download_puzzle_database(cache_dir)
    
    if not csv_path:
        return None, None, None
    
    positions, moves, values = process_puzzle_database(
        csv_path, max_positions, min_rating, max_rating,
        include_values, include_position_eval, themes_filter
    )
    
    if positions is not None:
        # Cache the processed data
        np.save(positions_cache, positions)
        np.save(moves_cache, moves)
        if include_position_eval and values is not None:
            np.save(values_cache, values)
        print(f"✓ Cached puzzle dataset to {cache_dir}")
        
        # Clean up CSV (it's huge - ~2GB)
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"✓ Cleaned up {os.path.basename(csv_path)}")
    
    return positions, moves, values


# Useful theme filters for different training goals
THEME_FILTERS = {
    "mate": ["mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"],
    "endgame": ["endgame", "queenEndgame", "rookEndgame", "bishopEndgame", "knightEndgame", "pawnEndgame"],
    "tactical": ["fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "sacrifice"],
    "advantage": ["advantage", "crushing", "hangingPiece", "trappedPiece"],
    "all_winning": None  # No filter, all puzzles are winning positions
}

