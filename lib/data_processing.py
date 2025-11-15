"""Data processing utilities for chess PGN files."""

import os
import subprocess
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from typing import Tuple, Optional


def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Convert chess board to 8x8x12 numpy array.
    Channels 0-5: white pieces (P,N,B,R,Q,K)
    Channels 6-11: black pieces (P,N,B,R,Q,K)
    
    Args:
        board: python-chess Board object
        
    Returns:
        8x8x12 numpy array
    """
    array = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            array[rank, file, channel] = 1.0
    
    return array


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess move to index.
    Index = from_square * 64 + to_square
    
    Args:
        move: python-chess Move object
        
    Returns:
        Integer index (0-4095)
    """
    return move.from_square * 64 + move.to_square


def download_and_extract_pgn(url: str, output_dir: str = ".") -> Optional[str]:
    """
    Download and extract a Lichess PGN database.
    
    Args:
        url: URL to .pgn.zst file
        output_dir: Directory to save files
        
    Returns:
        Path to extracted .pgn file, or None if failed
    """
    filename = url.split("/")[-1]
    zst_path = os.path.join(output_dir, filename)
    pgn_path = zst_path.replace(".zst", "")
    
    # Skip if already extracted
    if os.path.exists(pgn_path):
        print(f"✓ PGN already exists: {pgn_path}")
        return pgn_path
    
    try:
        # Download
        if not os.path.exists(zst_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-q", url, "-O", zst_path], check=True)
            print(f"✓ Downloaded {filename}")
        
        # Extract
        print(f"Extracting {filename}...")
        subprocess.run(["unzstd", zst_path], check=True)
        print(f"✓ Extracted to {pgn_path}")
        
        # Clean up compressed file
        if os.path.exists(zst_path):
            os.remove(zst_path)
            print(f"✓ Cleaned up {filename}")
        
        return pgn_path
        
    except Exception as e:
        print(f"✗ Failed to download/extract: {e}")
        return None


def process_pgn(
    pgn_path: str,
    max_positions: int = 500000,
    min_rating: int = 1800,
    skip_moves: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process PGN file and extract positions and moves.
    
    Args:
        pgn_path: Path to .pgn file
        max_positions: Maximum number of positions to extract
        min_rating: Minimum ELO rating for both players
        skip_moves: Number of opening moves to skip
        
    Returns:
        Tuple of (positions, moves) as numpy arrays, or (None, None) if failed
    """
    if not os.path.exists(pgn_path):
        print(f"✗ PGN file not found: {pgn_path}")
        return None, None
    
    positions_list = []
    moves_list = []
    
    print(f"\nProcessing {os.path.basename(pgn_path)}:")
    print(f"  Max positions: {max_positions}")
    print(f"  Min rating: {min_rating}")
    print(f"  Skip opening moves: {skip_moves}\n")
    
    try:
        with open(pgn_path, 'r') as pgn_file:
            games_processed = 0
            games_skipped = 0
            pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos")
            
            while len(positions_list) < max_positions:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    
                    if game is None:
                        break
                    
                    # Filter by rating
                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                        if white_elo < min_rating or black_elo < min_rating:
                            games_skipped += 1
                            continue
                    except (ValueError, TypeError):
                        games_skipped += 1
                        continue
                    
                    # Process game moves
                    board = game.board()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        move_count += 1
                        
                        # Skip opening moves
                        if move_count <= skip_moves:
                            board.push(move)
                            continue
                        
                        try:
                            positions_list.append(board_to_array(board))
                            moves_list.append(move_to_index(move))
                            pbar.update(1)
                            
                            if len(positions_list) >= max_positions:
                                break
                        except Exception:
                            pass
                        
                        board.push(move)
                    
                    games_processed += 1
                    if games_processed % 100 == 0:
                        pbar.set_postfix({
                            'games': games_processed,
                            'skipped': games_skipped
                        })
                
                except Exception:
                    continue
            
            pbar.close()
            print(f"\n✓ Processed {games_processed} games")
            print(f"  Skipped: {games_skipped} games")
            print(f"  Extracted: {len(positions_list)} positions")
    
    except Exception as e:
        print(f"✗ Error processing PGN: {e}")
        return None, None
    
    if not positions_list:
        print("✗ No positions extracted")
        return None, None
    
    positions = np.array(positions_list, dtype=np.float32)
    moves = np.array(moves_list, dtype=np.int32)
    
    print(f"✓ Final shapes: positions={positions.shape}, moves={moves.shape}")
    return positions, moves


def load_or_process_dataset(
    dataset_url: str,
    cache_dir: str,
    max_positions: int = 500000,
    min_rating: int = 1800,
    skip_moves: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cached dataset or download and process PGN.
    
    Args:
        dataset_url: URL to Lichess database
        cache_dir: Directory for caching processed data
        max_positions: Maximum positions to extract
        min_rating: Minimum ELO rating
        skip_moves: Opening moves to skip
        
    Returns:
        Tuple of (positions, moves) numpy arrays
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filenames based on URL
    dataset_name = dataset_url.split("/")[-1].replace(".pgn.zst", "")
    positions_cache = os.path.join(cache_dir, f"{dataset_name}_positions.npy")
    moves_cache = os.path.join(cache_dir, f"{dataset_name}_moves.npy")
    
    # Check cache
    if os.path.exists(positions_cache) and os.path.exists(moves_cache):
        print(f"✓ Loading cached dataset: {dataset_name}")
        positions = np.load(positions_cache)
        moves = np.load(moves_cache)
        print(f"  Loaded {len(positions)} positions")
        return positions, moves
    
    # Download and process
    print(f"Processing new dataset: {dataset_name}")
    pgn_path = download_and_extract_pgn(dataset_url, cache_dir)
    
    if not pgn_path:
        return None, None
    
    positions, moves = process_pgn(pgn_path, max_positions, min_rating, skip_moves)
    
    if positions is not None:
        # Cache the processed data
        np.save(positions_cache, positions)
        np.save(moves_cache, moves)
        print(f"✓ Cached dataset to {cache_dir}")
        
        # Clean up PGN file to save space
        if os.path.exists(pgn_path):
            os.remove(pgn_path)
            print(f"✓ Cleaned up {os.path.basename(pgn_path)}")
    
    return positions, moves
