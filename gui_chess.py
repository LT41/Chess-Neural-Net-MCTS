import pygame
import chess
import torch
import os
import numpy as np
import math
import threading  # Import threading module
from model import ChessNet
from utils import (
    DEVICE,
    POLICY_SIZE,
    encode_board,
    move_to_index,
    index_to_move,
    softmax,
)

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 512, 600  # Increased height for timer display
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess AI with Timer')

# Colors
LIGHT_SQUARE = pygame.Color(235, 236, 208)
DARK_SQUARE = pygame.Color(119, 154, 88)
HIGHLIGHT_COLOR = pygame.Color(186, 202, 68)
MOVE_HIGHLIGHT_COLOR = pygame.Color(214, 214, 189)
TEXT_COLOR = pygame.Color(0, 0, 0)
TIMER_BG_COLOR = pygame.Color(240, 240, 240)  # Light gray for timer background

# AI Parameters
NUM_SIMULATIONS = 1600
C_PUCT = 1.5
TEMPERATURE = 0.5

# Timer settings
INITIAL_TIME = 15 * 60  # 15 minutes in seconds
INCREMENT = 15  # 15 seconds

# Load piece images
PIECE_IMAGES = {}

def load_images():
    pieces = ['wp', 'wr', 'wn', 'wb', 'wq', 'wk',
              'bp', 'br', 'bn', 'bb', 'bq', 'bk']
    for piece in pieces:
        image_path = os.path.join('images', piece + '.png')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image for piece '{piece}' not found at '{image_path}'.")
        image = pygame.image.load(image_path)
        PIECE_IMAGES[piece] = pygame.transform.scale(image, (64, 64))

load_images()

def get_square_under_mouse():
    mouse_pos = pygame.mouse.get_pos()
    x, y = mouse_pos
    if x >= WIDTH or y >= 512:  # Ensure mouse is within the board area
        return None
    col = x // 64
    row = y // 64
    return chess.square(col, 7 - row)

def get_legal_moves(board, square):
    if square is None:
        return []
    legal_moves = []
    for move in board.legal_moves:
        if move.from_square == square:
            legal_moves.append(move.to_square)
    return legal_moves

def draw_board(window, board, selected_square=None, legal_moves=None):
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(window, color, pygame.Rect(file*64, rank*64, 64, 64))
            
            # Highlight selected square
            if selected_square == square:
                pygame.draw.rect(window, HIGHLIGHT_COLOR, pygame.Rect(file*64, rank*64, 64, 64))
            
            # Highlight legal moves
            if legal_moves and square in legal_moves:
                pygame.draw.rect(window, MOVE_HIGHLIGHT_COLOR, pygame.Rect(file*64, rank*64, 64, 64))
    
            piece = board.piece_at(square)
            if piece:
                piece_color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().lower()
                image_key = piece_color + piece_type
                if image_key in PIECE_IMAGES:
                    window.blit(PIECE_IMAGES[image_key], pygame.Rect(file*64, rank*64, 64, 64))
                else:
                    print(f"Missing image for piece: {image_key}")

def draw_timer(window, white_time, black_time):
    # Fill the timer area with a background color
    timer_rect = pygame.Rect(0, 512, WIDTH, HEIGHT - 512)
    pygame.draw.rect(window, TIMER_BG_COLOR, timer_rect)
    
    font = pygame.font.Font(None, 36)
    white_text = font.render(f"White: {int(white_time)//60:02d}:{int(white_time)%60:02d}", True, TEXT_COLOR)
    black_text = font.render(f"Black: {int(black_time)//60:02d}:{int(black_time)%60:02d}", True, TEXT_COLOR)
    
    # Display the timers
    window.blit(white_text, (10, 520))
    window.blit(black_text, (10, 560))

# Load your trained model
model = ChessNet().to(DEVICE)
model_path = 'merlin.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
state_dict = torch.load(model_path, map_location=DEVICE)['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = prior
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

def select_child(node):
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -float('inf')
    best_child = None

    for move, child in node.children.items():
        prior = child.prior
        q_value = child.value()
        u_value = C_PUCT * prior * math.sqrt(total_visits) / (1 + child.visit_count)
        score = q_value + u_value
        if score > best_score:
            best_score = score
            best_child = child

    return best_child

def evaluate(node):
    board_tensor = encode_board(node.board).to(DEVICE)

    with torch.no_grad():
        value, policy_logits = model(board_tensor.unsqueeze(0))
        value = value.item()
        policy_logits = policy_logits.squeeze(0).cpu().numpy()

    # Apply softmax to policy logits
    policy = softmax(policy_logits)

    # Get legal moves
    legal_moves = list(node.board.legal_moves)

    # Map moves to indices
    move_indices = []
    for move in legal_moves:
        idx = move_to_index(move)
        if idx is not None and idx < POLICY_SIZE:
            move_indices.append(idx)
        else:
            # Handle unmapped moves
            move_indices.append(None)

    # Normalize the policy for legal moves
    priors = []
    for move, idx in zip(legal_moves, move_indices):
        if idx is not None:
            priors.append(policy[idx])
        else:
            priors.append(0)

    priors = np.array(priors)
    if priors.sum() > 0:
        priors /= priors.sum()
    else:
        priors = np.ones(len(priors)) / len(priors)

    # Create child nodes
    for move, prior in zip(legal_moves, priors):
        board_copy = node.board.copy()
        board_copy.push(move)
        child_node = MCTSNode(board_copy, parent=node, prior=prior)
        node.children[move] = child_node

    node.is_expanded = True
    return value

def backpropagate(search_path, value):
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value if node.board.turn == chess.WHITE else -value
        value = -value  # Switch perspective

def get_ai_move(board, remaining_time):
    if board.is_game_over():
        return None

    root = MCTSNode(board.copy())
    
    start_time = pygame.time.get_ticks()
    elapsed_time = 0
    
    # Calculate maximum time to use for AI thinking
    max_thinking_time = remaining_time * 0.1  # Use up to 10% of remaining time
    max_thinking_time = min(max_thinking_time, 5)  # Cap to 5 seconds
    max_thinking_time = max(max_thinking_time, 0.1)  # Ensure at least 0.1 seconds

    while elapsed_time < max_thinking_time:
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded and node.children:
            node = select_child(node)
            search_path.append(node)

        # Expansion and Evaluation
        value = evaluate(node)

        # Backpropagation
        backpropagate(search_path, value)

        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000

    if not root.children:
        return None

    # Choose the move based on visit counts and temperature
    visits = np.array([child.visit_count for child in root.children.values()])
    if visits.sum() == 0:
        probabilities = np.ones(len(visits)) / len(visits)
    else:
        probabilities = softmax(1.0 / TEMPERATURE * np.log(visits + 1e-10))
    move = np.random.choice(list(root.children.keys()), p=probabilities)
    
    return move

# Global variables for AI threading
ai_move = None
ai_thinking_thread = None
ai_stop_event = threading.Event()
ai_lock = threading.Lock()

def ai_think(board_fen, remaining_time, stop_event):
    global ai_move
    board = chess.Board(board_fen)

    if board.is_game_over():
        with ai_lock:
            ai_move = None
        return

    root = MCTSNode(board.copy())

    start_time = pygame.time.get_ticks()
    elapsed_time = 0

    # Calculate maximum time to use for AI thinking
    max_thinking_time = remaining_time * 0.1  # Use up to 10% of remaining time
    max_thinking_time = min(max_thinking_time, 5)  # Cap to 5 seconds
    max_thinking_time = max(max_thinking_time, 0.1)  # Ensure at least 0.1 seconds

    while elapsed_time < max_thinking_time and not stop_event.is_set():
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded and node.children:
            node = select_child(node)
            search_path.append(node)

        # Expansion and Evaluation
        value = evaluate(node)

        # Backpropagation
        backpropagate(search_path, value)

        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000

    if not root.children:
        with ai_lock:
            ai_move = None
        return

    # Choose the move based on visit counts and temperature
    visits = np.array([child.visit_count for child in root.children.values()])
    if visits.sum() == 0:
        probabilities = np.ones(len(visits)) / len(visits)
    else:
        probabilities = softmax(1.0 / TEMPERATURE * np.log(visits + 1e-10))
    move = np.random.choice(list(root.children.keys()), p=probabilities)

    with ai_lock:
        ai_move = move

def main():
    global ai_move, ai_thinking_thread, ai_stop_event
    board = chess.Board()
    selected_square = None
    legal_moves = []
    move_made = False
    running = True
    clock = pygame.time.Clock()
    player_color = chess.WHITE  # You can set this to chess.BLACK to play as black

    white_time = INITIAL_TIME
    black_time = INITIAL_TIME
    last_move_time = pygame.time.get_ticks()

    while running:
        # Handle events first to minimize input lag
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == player_color:
                square = get_square_under_mouse()
                if square is not None:
                    piece = board.piece_at(square)
                    if selected_square is None:
                        # First click
                        if piece and piece.color == player_color:
                            selected_square = square
                            legal_moves = get_legal_moves(board, square)
                    else:
                        if square == selected_square:
                            # Clicked the same square again, deselect
                            selected_square = None
                            legal_moves = []
                        elif piece and piece.color == player_color:
                            # Clicked another own piece, select it
                            selected_square = square
                            legal_moves = get_legal_moves(board, square)
                        else:
                            # Attempt to make a move
                            move = chess.Move(selected_square, square)
                            if move in board.legal_moves:
                                board.push(move)
                                move_made = True
                                if player_color == chess.WHITE:
                                    white_time += INCREMENT
                                else:
                                    black_time += INCREMENT
                                selected_square = None
                                legal_moves = []
                                last_move_time = pygame.time.get_ticks()
                                # Stop AI thinking when player makes a move
                                ai_stop_event.set()
                                if ai_thinking_thread and ai_thinking_thread.is_alive():
                                    ai_thinking_thread.join()
                                ai_stop_event.clear()
                                ai_move = None
                            else:
                                # Invalid move or clicked empty square, deselect
                                selected_square = None
                                legal_moves = []
                else:
                    # Clicked outside the board, deselect
                    selected_square = None
                    legal_moves = []

        # Start AI thinking during player's turn
        if board.turn == player_color and not board.is_game_over():
            if not ai_thinking_thread or not ai_thinking_thread.is_alive():
                ai_stop_event.clear()
                ai_thinking_thread = threading.Thread(
                    target=ai_think,
                    args=(board.fen(), black_time if board.turn == chess.BLACK else white_time, ai_stop_event)
                )
                ai_thinking_thread.start()

        # AI's turn
        if board.turn != player_color and not board.is_game_over():
            ai_start_time = pygame.time.get_ticks()

            # Wait for AI thinking thread to finish if necessary
            if ai_thinking_thread and ai_thinking_thread.is_alive():
                ai_stop_event.set()
                ai_thinking_thread.join()
                ai_stop_event.clear()

            ai_end_time = pygame.time.get_ticks()
            ai_thinking_time = (ai_end_time - ai_start_time) / 1000.0

            # Subtract AI's thinking time from its clock
            if board.turn == chess.BLACK:
                black_time -= ai_thinking_time
                black_time = max(black_time, 0)
            else:
                white_time -= ai_thinking_time
                white_time = max(white_time, 0)

            # Use the precomputed move
            with ai_lock:
                move = ai_move

            if move is not None:
                board.push(move)
                print(f"AI played: {move}")
                if board.turn == chess.WHITE:
                    black_time += INCREMENT
                else:
                    white_time += INCREMENT
            else:
                # If no precomputed move, compute one now
                ai_remaining_time = black_time if board.turn == chess.BLACK else white_time
                move = get_ai_move(board, ai_remaining_time)
                if move is not None:
                    board.push(move)
                    print(f"AI played: {move}")
                    if board.turn == chess.WHITE:
                        black_time += INCREMENT
                    else:
                        white_time += INCREMENT
                else:
                    print("AI has no legal moves.")

            # Reset AI move and stop event
            ai_move = None
            ai_stop_event.clear()
            ai_thinking_thread = None

            # Update last_move_time after AI's move
            last_move_time = pygame.time.get_ticks()

        # Update timers
        current_time = pygame.time.get_ticks()
        delta_time = (current_time - last_move_time) / 1000.0

        if board.turn == chess.WHITE:
            white_time -= delta_time
            white_time = max(white_time, 0)
        else:
            black_time -= delta_time
            black_time = max(black_time, 0)

        # Reset last_move_time
        last_move_time = current_time

        # Draw the board and timer
        draw_board(WINDOW, board, selected_square, legal_moves)
        draw_timer(WINDOW, white_time, black_time)
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS

        # Check for game over
        if board.is_game_over() or white_time <= 0 or black_time <= 0:
            if board.is_game_over():
                result = board.result()
                if board.is_checkmate():
                    winner = "White" if board.turn == chess.BLACK else "Black"
                    print(f"Checkmate! {winner} wins.")
                elif board.is_stalemate():
                    print("Stalemate!")
                elif board.is_insufficient_material():
                    print("Insufficient material!")
                elif board.can_claim_fifty_moves():
                    print("Fifty-move rule!")
                elif board.can_claim_threefold_repetition():
                    print("Threefold repetition!")
                else:
                    print("Game over:", result)
            elif white_time <= 0:
                print("Black wins on time")
            elif black_time <= 0:
                print("White wins on time")
            running = False

    pygame.quit()

if __name__ == '__main__':
    main()
