import math
from copy import deepcopy

import numpy as np

from .utils import chess_manager, GameContext
from chess import Move
import random

import random
from math import log, sqrt

class Node:
    def __init__(self, state):
        self.state = state
        self.win_value = 0
        self.policy_value = None
        self.visits = 0
        self.parent = None
        self.children = []
        self.expanded = False
        self.player_number = None
        self.discovery_factor = 0.35

    def update_win_value(self, value):
        self.win_value += value
        self.visits += 1

        if self.parent:
            self.parent.update_win_value(value)

    def update_policy_value(self, value):
        self.policy_value = value

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_preferred_child(self, root_node):
        best_children = []
        best_score = float('-inf')

        for child in self.children:
            score = child.get_score(root_node)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def get_score(self, root_node):
        discovery_operand = self.discovery_factor * (self.policy_value or 1) * sqrt(log(self.parent.visits or 1) / (self.visits or 1))
        win_multiplier = 1 if self.parent.player_number == root_node.player_number else -1
        win_operand = win_multiplier * self.win_value / (self.visits or 1)
        self.score = win_operand + discovery_operand

        self.score += len(list(self.state.generate_legal_moves()))

        pawns = self.state.pawns.bit_count() * 1
        knights = self.state.knights.bit_count() * 3
        bishops = self.state.bishops.bit_count() * 3
        rooks = self.state.rooks.bit_count() * 5
        queens = self.state.queens.bit_count() * 9

        piece_score = pawns + knights + bishops + rooks + queens

        return self.score + piece_score / 40.0

    def is_scorable(self):
        return self.visits or self.policy_value != None

class MonteCarlo:

    def __init__(self, root_node):
        self.root_node = root_node
        self.child_finder = None
        self.node_evaluator = lambda child, montecarlo: int

    def make_choice(self):
        best_children = []
        most_visits = float('-inf')

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def make_exploratory_choice(self):
        children_visits = map(lambda child: child.visits, self.root_node.children)
        children_visit_probabilities = [visit / self.root_node.visits for visit in children_visits]
        random_probability = random.uniform(0, sum(children_visit_probabilities))
        probabilities_already_counted = 0.

        for i, probability in enumerate(children_visit_probabilities):
            if probabilities_already_counted + probability >= random_probability:
                return self.root_node.children[i]

            probabilities_already_counted += probability

    def simulate(self, expansion_count = 1):
        for i in range(expansion_count):
            current_node = self.root_node
            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)

    def expand(self, node):
        self.child_finder(node, self)

        for child in node.children:
            child_win_value = self.node_evaluator(child, self)

            if child_win_value != None:
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True

    def random_rollout(self, node):
        self.child_finder(node, self)
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        child_win_value = self.node_evaluator(child, self)

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            self.random_rollout(child)

# Load the trained LC0 neural network model
print("🤖 Loading LC0 model from HuggingFace...")
from .models.lc0_inference import LC0ModelLoader

model_loader = LC0ModelLoader(
    repo_id="steveandcow/chesshacks-lc0",
    model_file="lc0_128x6.pt",
    device="cpu"  # Use CPU for deployment (or "cuda" if GPU available)
)

try:
    model_loader.load_model()
    print("✅ LC0 Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    print("Falling back to random policy...")
    import traceback
    traceback.print_exc()
    MODEL_LOADED = False

def child_finder(node, montecarlo):
    for move in node.state.generate_legal_moves():
        new_board = deepcopy(node.state) # Note, probably slow
        new_board.push(move)
        child = Node(new_board)
        node.add_child(child)

def node_evaluator(node, montecarlo):
    if node.state.is_variant_win():
        return math.inf
    elif node.state.is_variant_loss():
        return -math.inf

    # Use neural network for position evaluation if available
    if MODEL_LOADED:
        try:
            value = model_loader.evaluate_position(node.state)
            return value
        except Exception as e:
            print(f"⚠️ NN evaluation failed: {e}")
            pass

    return node.get_score(montecarlo.root_node)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Get move probabilities from neural network
    if MODEL_LOADED:
        try:
            move_probs, position_value = model_loader.predict(ctx.board)
            print(f"Position value: {position_value:.3f}")

            # Log probabilities for analysis
            ctx.logProbabilities(move_probs)

            # ========================================
            # ADAPTIVE TIME MANAGEMENT
            # Uses MCTS when safe, NN-only when tight
            # ========================================
            time_left_ms = ctx.timeLeft
            print(f"⏱️  Time remaining: {time_left_ms}ms")

            # Strategy: Reserve 300ms overhead, use rest for search
            # Each MCTS simulation costs ~100ms (NN inference + tree ops)
            OVERHEAD_MS = 300
            MS_PER_SIMULATION = 100

            # Calculate safe simulation budget
            available_time = max(time_left_ms - OVERHEAD_MS, 0)
            max_simulations = available_time // MS_PER_SIMULATION

            # Decision thresholds
            MIN_TIME_FOR_MCTS = 800  # Need at least 800ms for any MCTS
            MIN_SIMULATIONS = 3      # Minimum sims if doing MCTS
            MAX_SIMULATIONS = 16     # Cap to avoid runaway

            # Decide strategy
            if time_left_ms < MIN_TIME_FOR_MCTS or max_simulations < MIN_SIMULATIONS:
                # FAST MODE: Pure NN policy (~100ms)
                print(f"⚡ FAST MODE: Using NN policy (insufficient time for MCTS)")
                best_move = max(move_probs.items(), key=lambda x: x[1])[0]
                return best_move

            # SEARCH MODE: Use MCTS with limited simulations
            num_simulations = min(max_simulations, MAX_SIMULATIONS)
            print(f"🎯 SEARCH MODE: Running {num_simulations} MCTS simulations")

            montecarlo = MonteCarlo(Node(ctx.board))
            montecarlo.child_finder = child_finder
            montecarlo.node_evaluator = node_evaluator
            montecarlo.simulate(num_simulations)

            # Get best move from MCTS
            best_child = montecarlo.make_choice()
            best_move = None
            for move in legal_moves:
                test_board = deepcopy(ctx.board)
                test_board.push(move)
                if test_board.fen() == best_child.state.fen():
                    best_move = move
                    break

            if best_move:
                print(f"✓ MCTS selected move: {best_move.uci()}")
                return best_move

            # Fallback: use NN policy directly
            print("⚠️ MCTS fallback to NN policy")
            return max(move_probs.items(), key=lambda x: x[1])[0]

        except Exception as e:
            print(f"⚠️ NN prediction failed: {e}")
            import traceback
            traceback.print_exc()

    # Fallback to random if NN not available
    print("⚠️ Using random policy")
    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    move_probs = {move: weight / total_weight for move, weight in zip(legal_moves, move_weights)}
    ctx.logProbabilities(move_probs)
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass