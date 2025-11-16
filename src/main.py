import math
from copy import deepcopy
import time

import numpy as np

from .utils import chess_manager, GameContext
from chess import Move
import random

import random
from math import log, sqrt

class Node:
    def __init__(self, state, move=None):
        self.state = state
        self.move = move  # Store the move that led to this state (for O(1) lookup)
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
        self.early_stop_threshold = 0.7  # Stop if best move has 70% of visits

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

    def should_stop_early(self):
        """Early stopping: if one move is clearly dominant, stop searching."""
        if len(self.root_node.children) < 2:
            return True

        sorted_children = sorted(self.root_node.children, key=lambda c: c.visits, reverse=True)
        best_visits = sorted_children[0].visits
        second_visits = sorted_children[1].visits if len(sorted_children) > 1 else 0

        # If best move has threshold% of visits, it's clearly winning
        if self.root_node.visits > 5 and best_visits > self.early_stop_threshold * self.root_node.visits:
            return True

        return False

    def simulate(self, expansion_count = 1):
        for i in range(expansion_count):
            current_node = self.root_node
            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)

            # Early stopping check
            if self.should_stop_early():
                print(f"🎯 Early stop after {i+1}/{expansion_count} simulations")
                break

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

# Transposition table for caching NN evaluations
position_cache = {}
cache_hits = 0
cache_misses = 0

def child_finder(node, montecarlo):
    """Generate child nodes for all legal moves. OPTIMIZED: uses board.copy() instead of deepcopy."""
    for move in node.state.legal_moves:
        # Use board.copy() which is much faster than deepcopy (~1ms vs ~10ms)
        new_board = node.state.copy()
        new_board.push(move)
        child = Node(new_board, move=move)  # Store move for O(1) lookup
        node.add_child(child)

def node_evaluator(node, montecarlo):
    """Evaluate node value using NN with transposition table caching."""
    global cache_hits, cache_misses

    if node.state.is_variant_win():
        return math.inf
    elif node.state.is_variant_loss():
        return -math.inf

    # Use neural network for position evaluation if available
    if MODEL_LOADED:
        try:
            # Check transposition table first
            fen = node.state.fen()
            if fen in position_cache:
                cache_hits += 1
                return position_cache[fen]

            cache_misses += 1
            value = model_loader.evaluate_position(node.state)
            position_cache[fen] = value  # Cache for future lookups
            return value
        except Exception as e:
            print(f"⚠️ NN evaluation failed: {e}")
            pass

    return node.get_score(montecarlo.root_node)


def calculate_move_budget(board, time_left_ms):
    """
    Dynamic time allocation based on game phase and position complexity.
    """
    move_number = board.fullmove_number
    total_time = time_left_ms

    # Estimate moves remaining
    if move_number < 15:
        estimated_moves_left = 60  # Opening
    elif move_number < 40:
        estimated_moves_left = 50 - move_number  # Middlegame
    else:
        estimated_moves_left = 20  # Endgame

    # Base allocation: divide remaining time evenly
    base_time = total_time / max(estimated_moves_left, 1)

    # Adjust for position complexity
    num_legal_moves = board.legal_moves.count()
    if num_legal_moves < 10:
        complexity_factor = 0.5  # Simple position, use less time
    elif num_legal_moves > 30:
        complexity_factor = 2.0  # Complex position, use more time
    else:
        complexity_factor = 1.0

    # Critical positions (checks, captures available)
    if board.is_check():
        complexity_factor *= 1.5

    # Safety margin: never use more than 20% of remaining time on one move
    max_time = min(base_time * complexity_factor, total_time * 0.2)

    # Minimum time: always do at least quick NN evaluation
    return max(max_time, 50)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main move generation function with performance optimizations and profiling.
    """
    global cache_hits, cache_misses

    # ========================================
    # PROFILING: Start timing
    # ========================================
    move_start_time = time.time()
    timings = {}

    print("\n" + "="*60)
    print(f"🎮 Move {ctx.board.fullmove_number} ({'White' if ctx.board.turn else 'Black'})")
    print("="*60)

    t0 = time.time()
    legal_moves = list(ctx.board.legal_moves)
    timings['legal_moves_gen'] = (time.time() - t0) * 1000

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    print(f"Legal moves: {len(legal_moves)}")
    print(f"Position: {ctx.board.fen()[:50]}...")

    # Get move probabilities from neural network
    if MODEL_LOADED:
        try:
            t0 = time.time()
            move_probs, position_value = model_loader.predict(ctx.board)
            timings['nn_predict'] = (time.time() - t0) * 1000

            print(f"📊 Position value: {position_value:.3f}")

            # Log probabilities for analysis
            ctx.logProbabilities(move_probs)

            # ========================================
            # DYNAMIC TIME BUDGET CALCULATION
            # ========================================
            time_left_ms = ctx.timeLeft
            t0 = time.time()
            allocated_time_ms = calculate_move_budget(ctx.board, time_left_ms)
            timings['time_budget_calc'] = (time.time() - t0) * 1000

            print(f"⏱️  Time left: {time_left_ms}ms | Allocated: {allocated_time_ms:.0f}ms")

            # Strategy: Each MCTS simulation costs ~20-50ms now (with optimizations)
            MS_PER_SIMULATION = 30  # Updated estimate with optimizations
            OVERHEAD_MS = 100  # Reduced overhead

            # Calculate safe simulation budget
            available_time = max(allocated_time_ms - OVERHEAD_MS, 0)
            max_simulations = int(available_time // MS_PER_SIMULATION)

            # Decision thresholds
            MIN_TIME_FOR_MCTS = 200  # Reduced threshold (was 800ms)
            MIN_SIMULATIONS = 3
            MAX_SIMULATIONS = 200  # Increased cap (was 16)

            # Decide strategy
            if time_left_ms < MIN_TIME_FOR_MCTS or max_simulations < MIN_SIMULATIONS:
                # FAST MODE: Pure NN policy
                print(f"⚡ FAST MODE: Using NN policy (insufficient time for MCTS)")
                best_move = max(move_probs.items(), key=lambda x: x[1])[0]

                # Final profiling
                total_time = (time.time() - move_start_time) * 1000
                print(f"\n⏱️  TOTAL TIME: {total_time:.1f}ms")
                return best_move

            # SEARCH MODE: Use MCTS with limited simulations
            num_simulations = min(max_simulations, MAX_SIMULATIONS)
            print(f"🔍 SEARCH MODE: Running up to {num_simulations} MCTS simulations")

            t0 = time.time()
            montecarlo = MonteCarlo(Node(ctx.board))
            montecarlo.child_finder = child_finder
            montecarlo.node_evaluator = node_evaluator

            # Reset cache stats for this move
            old_cache_hits = cache_hits
            old_cache_misses = cache_misses

            montecarlo.simulate(num_simulations)
            timings['mcts_search'] = (time.time() - t0) * 1000

            # Get best move from MCTS (OPTIMIZED: O(1) lookup using stored move)
            t0 = time.time()
            best_child = montecarlo.make_choice()
            best_move = best_child.move  # O(1) lookup! No more deepcopy loop
            timings['best_move_select'] = (time.time() - t0) * 1000

            # Cache statistics for this move
            move_cache_hits = cache_hits - old_cache_hits
            move_cache_misses = cache_misses - old_cache_misses
            cache_hit_rate = move_cache_hits / max(move_cache_hits + move_cache_misses, 1) * 100

            if best_move:
                print(f"✅ MCTS selected: {best_move.uci()}")
                print(f"   Visits: {best_child.visits} | Value: {best_child.win_value / max(best_child.visits, 1):.3f}")
                print(f"   Cache: {move_cache_hits} hits, {move_cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")

                # Final profiling
                total_time = (time.time() - move_start_time) * 1000
                print(f"\n⏱️  TIMING BREAKDOWN:")
                print(f"   Legal moves gen:  {timings['legal_moves_gen']:6.1f}ms")
                print(f"   NN prediction:    {timings['nn_predict']:6.1f}ms")
                print(f"   Time budget calc: {timings['time_budget_calc']:6.1f}ms")
                print(f"   MCTS search:      {timings['mcts_search']:6.1f}ms")
                print(f"   Best move select: {timings['best_move_select']:6.1f}ms")
                print(f"   ────────────────────────────")
                print(f"   TOTAL:            {total_time:6.1f}ms")
                print(f"   Budget remaining: {allocated_time_ms - total_time:6.1f}ms")

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

    total_time = (time.time() - move_start_time) * 1000
    print(f"\n⏱️  TOTAL TIME: {total_time:.1f}ms")

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Clear caches and reset state.
    """
    global position_cache, cache_hits, cache_misses

    print("\n🔄 RESET: New game starting")
    print(f"   Cache stats from last game: {cache_hits} hits, {cache_misses} misses")

    # Clear transposition table
    position_cache.clear()
    cache_hits = 0
    cache_misses = 0

    print("   Cache cleared ✓")