import torch
import torch.nn.functional as F
import numpy as np
import chess
import os
import pickle
import gzip
import random
from tqdm import tqdm

# Configuration Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POLICY_SIZE = 4672  # Should match the output size of your model's policy head
C_PUCT = 1.0        # Exploration constant in MCTS
TEMPERATURE = 1.0   # Controls exploration
NUM_SIMULATIONS = 200  # Number of MCTS simulations per move

def encode_board(board):
    """
    Encodes the board into a 14x8x8 PyTorch tensor.
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        piece_type = piece.symbol()
        plane = piece_map.get(piece_type, None)
        if plane is not None:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[plane][row][col] = 1

    # Turn
    tensor[12] = torch.ones((8,8)) if board.turn == chess.WHITE else torch.zeros((8,8))

    # No-op plane (can be used for additional information)
    tensor[13] = torch.zeros((8,8))

    return tensor

def move_to_index(move):
    """
    Encodes a move into an index based on custom move encoding.
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    # Calculate move vector
    delta = to_square - from_square
    row_diff = chess.square_rank(to_square) - chess.square_rank(from_square)
    col_diff = chess.square_file(to_square) - chess.square_file(from_square)

    # Determine direction
    direction = None
    if row_diff > 0 and col_diff == 0:
        direction = 0  # Up
    elif row_diff < 0 and col_diff == 0:
        direction = 1  # Down
    elif row_diff == 0 and col_diff > 0:
        direction = 2  # Right
    elif row_diff == 0 and col_diff < 0:
        direction = 3  # Left
    elif row_diff > 0 and col_diff > 0:
        direction = 4  # Up-Right
    elif row_diff > 0 and col_diff < 0:
        direction = 5  # Up-Left
    elif row_diff < 0 and col_diff > 0:
        direction = 6  # Down-Right
    elif row_diff < 0 and col_diff < 0:
        direction = 7  # Down-Left

    if direction is not None:
        # Calculate distance (1-7)
        distance = max(abs(row_diff), abs(col_diff)) - 1  # Zero-based index

        index = from_square * 56 + direction * 7 + distance
        return index
    else:
        # Handle knight moves or castling
        # Assign unique indices for these moves
        # For simplicity, we can assign them after the main indices
        special_move_offset = 8 * 7 * 64
        if promotion:
            # Promotion moves
            promotion_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
            promotion_code = promotion_map.get(promotion, 0)
            index = special_move_offset + from_square * 4 + promotion_code
            return index
        else:
            # Other special moves (e.g., knight moves, castling)
            # Assign indices after promotions
            knight_move_offset = special_move_offset + 64 * 4
            index = knight_move_offset + from_square * 8 + (delta % 8)
            return index

def index_to_move(index, board):
    """
    Decodes an index back to a chess.Move in the given board position.
    """
    legal_moves = list(board.legal_moves)
    move_map = {move_to_index(move): move for move in legal_moves}
    move = move_map.get(index)
    return move

def create_legal_move_indices(board):
    """
    Creates a list of legal move indices for the given board.
    """
    legal_moves = list(board.legal_moves)
    legal_indices = []
    for move in legal_moves:
        idx = move_to_index(move)
        if idx is not None and idx < POLICY_SIZE:
            legal_indices.append(idx)
    return legal_indices

class Node:
    def __init__(self, board, parent=None, prior=1.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior  # P(s, a)
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct):
        """
        Calculates the UCB score for the node.
        """
        return self.value() + c_puct * self.prior * (np.sqrt(self.parent.visit_count) / (1 + self.visit_count))

def evaluate_position(model, board):
    """
    Evaluates the board position using the neural network.
    """
    tensor = encode_board(board)
    X = tensor.unsqueeze(0).to(DEVICE)  # Shape: (1,14,8,8)
    with torch.no_grad():
        predicted_value, policy_logits = model(X)
        policy_logits = policy_logits.cpu().squeeze(0)

        # Mask illegal moves
        legal_indices = create_legal_move_indices(board)
        mask = torch.full_like(policy_logits, float('-inf'))
        mask[legal_indices] = 0
        masked_policy_logits = policy_logits + mask

        # Apply softmax to get probabilities
        policy_probs = F.softmax(masked_policy_logits, dim=0).numpy()

    return predicted_value.item(), policy_probs

def expand_node(model, node):
    """
    Expands the node by adding all legal moves as children.
    """
    board = node.board
    if board.is_game_over():
        node.is_expanded = True
        return

    value, policy_probs = evaluate_position(model, board)

    legal_moves = list(board.legal_moves)
    legal_indices = create_legal_move_indices(board)
    total_prob = np.sum(policy_probs[legal_indices]) + 1e-8  # Prevent division by zero

    for move in legal_moves:
        idx = move_to_index(move)
        if idx is None or idx >= POLICY_SIZE:
            continue
        prob = policy_probs[idx] / total_prob
        new_board = board.copy()
        new_board.push(move)
        child_node = Node(new_board, parent=node, prior=prob)
        node.children[move] = child_node

    node.is_expanded = True

def simulate(model, node):
    """
    Uses the value network to evaluate the board state.
    """
    value, _ = evaluate_position(model, node.board)
    return value

def backpropagate(path, value):
    """
    Updates the nodes' statistics with the simulation result.
    """
    for node in reversed(path):
        node.visit_count += 1
        node.total_value += value
        value = -value  # Switch perspective for the opponent

def mcts_search(model, root, num_simulations, min_simulations, early_stop_threshold):
    """
    Performs MCTS starting from the root node with early stopping.
    """
    for i in range(num_simulations):
        node = root
        path = [node]
        # Selection
        while node.is_expanded and node.children:
            # Select child with highest UCB score
            node = max(node.children.values(), key=lambda n: n.ucb_score(C_PUCT))
            path.append(node)

        # Expansion
        if not node.is_expanded:
            expand_node(model, node)

        # Simulation
        value = simulate(model, node)

        # Backpropagation
        backpropagate(path, value)

        # Early stopping check
        if i >= min_simulations:
            best_child = max(root.children.values(), key=lambda n: n.visit_count)
            visit_ratio = best_child.visit_count / (i + 1)
            if visit_ratio > early_stop_threshold:
                break

def mcts_policy(model, board, num_simulations, min_simulations, early_stop_threshold):
    """
    Performs MCTS to get the move probabilities (policy) with early stopping.
    """
    root = Node(board)
    mcts_search(model, root, num_simulations, min_simulations, early_stop_threshold)

    # Build the policy vector
    policy = np.zeros(POLICY_SIZE)
    total_visits = sum(child.visit_count for child in root.children.values())

    if total_visits == 0:
        # Avoid division by zero; assign uniform probabilities over legal moves
        legal_indices = create_legal_move_indices(board)
        for idx in legal_indices:
            policy[idx] = 1 / len(legal_indices)
    else:
        for move, child in root.children.items():
            idx = move_to_index(move)
            if idx is not None and idx < POLICY_SIZE:
                policy[idx] = child.visit_count / total_visits

    # Apply temperature
    if TEMPERATURE > 0:
        policy = policy ** (1 / TEMPERATURE)
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
    else:
        # Deterministic: pick the move with the highest probability
        best_moves = np.where(policy == np.max(policy))[0]
        policy = np.zeros_like(policy)
        policy[best_moves] = 1 / len(best_moves)

    return policy

def normalize_evaluation(eval_cp):
    # Assume evaluations are between -1000 and +1000 centipawns
    normalized = eval_cp / 1000.0
    normalized = max(min(normalized, 1), -1)  # Clamp between -1 and 1
    return normalized

def softmax(x):
    x = x - np.max(x)  # For numerical stability
    e_x = np.exp(x)
    return e_x / e_x.sum()