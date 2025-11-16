import math
from copy import deepcopy

import numpy as np

from .utils import chess_manager, GameContext
from chess import Move
import chess
import random

import random
from math import log, sqrt

import math
import random
from typing import Optional, Callable

class Node:
    __slots__ = (
        "state",
        "win_value",
        "policy_value",
        "visits",
        "parent",
        "children",
        "expanded",
        "player_number",
        "discovery_factor",
        "_cached_piece_score",
        "_cached_legal_moves_count",
        "score",  # optional cached score
        "virtual_loss",  # for batch inference
    )

    def __init__(self, state):
        self.state = state
        self.win_value = 0.0
        self.policy_value: Optional[float] = None
        self.visits = 0
        self.parent: Optional["Node"] = None
        self.children: list["Node"] = []
        self.expanded = False
        self.player_number = None
        self.discovery_factor = 0.35
        # caches to avoid repeated expensive work
        self._cached_piece_score: Optional[float] = None
        self._cached_legal_moves_count: Optional[int] = None
        self.score = 0.0
        self.virtual_loss = 0  # virtual loss for batch inference

    def update_win_value_iterative(self, value: float):
        """Iteratively propagate win_value and visits up to the root (fast, no recursion)."""
        node = self
        while node is not None:
            node.win_value += value
            node.visits += 1
            node = node.parent

    # keep compatibility naming: update_win_value used in existing code
    update_win_value = update_win_value_iterative

    def update_policy_value(self, value: float):
        self.policy_value = float(value)

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)

    def add_children(self, children: list["Node"]):
        # inline loop is fastest in Python, avoid additional function calls
        for c in children:
            c.parent = self
            self.children.append(c)

    def _compute_piece_score(self) -> float:
        """Compute and cache a simple material heuristic from the state's bitboards.
           If the state already supplies a fast method, override this or set the cache externally."""
        if self._cached_piece_score is not None:
            return self._cached_piece_score

        s = self.state
        # Expect that these are python ints (bitboards) with bit_count available.
        # If state structure differs, replace this block with a faster accessor.
        pawns = s.pawns.bit_count() if hasattr(s, "pawns") else 0
        knights = s.knights.bit_count() if hasattr(s, "knights") else 0
        bishops = s.bishops.bit_count() if hasattr(s, "bishops") else 0
        rooks = s.rooks.bit_count() if hasattr(s, "rooks") else 0
        queens = s.queens.bit_count() if hasattr(s, "queens") else 0

        piece_score = pawns + knights * 3 + bishops * 3 + rooks * 5 + queens * 9
        # store normalized or raw; keep raw and divide later for tunability
        self._cached_piece_score = float(piece_score)
        return self._cached_piece_score

    def _compute_legal_moves_count(self) -> int:
        if self._cached_legal_moves_count is not None:
            return self._cached_legal_moves_count
        # use generator consumption (no list allocation)
        try:
            gen = self.state.generate_legal_moves()
            count = 0
            for _ in gen:
                count += 1
        except Exception:
            # fallback if state uses list or other API
            try:
                lm = self.state.generate_legal_moves()
                count = len(lm)
            except Exception:
                count = 0
        self._cached_legal_moves_count = count
        return count

    def get_score(self, root_node: "Node") -> float:
        """Fast UCT-like score with cached piece/legals. Avoid repeated attribute lookups."""
        parent = self.parent
        if parent is None:
            # root child without parent should be handled elsewhere; return a large value to prefer others
            return float("inf")

        # local copies
        visits = self.visits
        parent_visits = parent.visits

        # Account for virtual loss - makes nodes being evaluated less attractive
        effective_visits = visits + self.virtual_loss

        # avoid zero division
        visits_safe = effective_visits if effective_visits > 0 else 1
        parent_visits_safe = parent_visits if parent_visits > 0 else 1

        discovery_factor = self.discovery_factor
        policy = self.policy_value if (self.policy_value is not None) else 1.0

        # localize math functions
        log_parent = math.log(parent_visits_safe)
        sqrt_term = math.sqrt(log_parent / visits_safe)
        discovery_operand = discovery_factor * policy * sqrt_term

        # win operand uses sign depending on ownership relative to root player
        # Virtual loss penalty: assume losses for nodes being evaluated
        effective_win_value = self.win_value - self.virtual_loss
        win_multiplier = 1.0 if parent.player_number == root_node.player_number else -1.0
        win_operand = win_multiplier * (effective_win_value / visits_safe)

        # small heuristic features
        legal_count = self._compute_legal_moves_count()
        piece_score = self._compute_piece_score()

        # final combined score; piece_score scaled down
        s = win_operand + discovery_operand + legal_count + (piece_score / 40.0)

        # cache last computed score (useful if selection queries repeatedly)
        self.score = s
        return s

    def get_preferred_child(self, root_node: "Node") -> "Node":
        """Return one of the children with highest score. Use max to avoid Python loops where possible,
           then break ties by selecting randomly among equals."""
        children = self.children
        if not children:
            raise RuntimeError("No children to select from")

        # compute scores and find max score
        best_score = float("-inf")
        best_children = []
        # inline loop and localize function for speed
        get_score = Node.get_score
        for c in children:
            sc = get_score(c, root_node)
            if sc > best_score:
                best_score = sc
                best_children = [c]
            elif sc == best_score:
                best_children.append(c)

        # random tie-break (rare)
        return random.choice(best_children)

    def is_scorable(self) -> bool:
        # quicker boolean checks
        return (self.visits > 0) or (self.policy_value is not None)

    def add_virtual_loss(self):
        """Add virtual loss to prevent multiple simulations from selecting the same node."""
        node = self
        while node is not None:
            node.virtual_loss += 1
            node = node.parent

    def remove_virtual_loss(self):
        """Remove virtual loss after evaluation is complete."""
        node = self
        while node is not None:
            node.virtual_loss = max(0, node.virtual_loss - 1)
            node = node.parent


class MonteCarlo:
    __slots__ = ("root_node", "child_finder", "node_evaluator", "batch_evaluator")

    def __init__(self, root_node: Node):
        self.root_node = root_node
        # Set these externally - child_finder(node, montecarlo) populates node.children
        self.child_finder: Callable[[Node, "MonteCarlo"], None] = lambda node, mc: None
        # node_evaluator(child, mc) -> optional numeric score (win_value) or None
        self.node_evaluator: Callable[[Node, "MonteCarlo"], Optional[float]] = lambda child, mc: None
        # batch_evaluator([nodes]) -> [values] for batched evaluation
        self.batch_evaluator: Optional[Callable[[list], list]] = None

    def make_choice(self) -> Node:
        """Return child with most visits (ties broken uniformly)."""
        children = self.root_node.children
        if not children:
            raise RuntimeError("No children available")
        # find maximum visits
        max_visits = max(c.visits for c in children)
        # collect ties
        best = [c for c in children if c.visits == max_visits]
        return random.choice(best)

    def make_exploratory_choice(self) -> Node:
        """Sample a child proportionally to visits. Works even if sum != 1."""
        children = self.root_node.children
        total = sum(c.visits for c in children)
        if total == 0:
            # fallback uniform if no visits yet
            return random.choice(children)
        r = random.uniform(0.0, total)
        acc = 0.0
        for c in children:
            acc += c.visits
            if acc >= r:
                return c
        return children[-1]  # safety fallback

    def should_stop_early(self) -> bool:
        """
        Check if we should stop MCTS early due to obvious best move.

        This implements more aggressive early stopping than before:
        - If one move has 80%+ of visits, stop
        - If best move has 3x more visits than second best, stop
        - Requires at least 10 total visits to prevent premature stopping
        """
        if not self.root_node.children:
            return False

        if len(self.root_node.children) < 2:
            return True  # Only one legal move

        total_visits = self.root_node.visits
        if total_visits < 10:
            return False  # Need minimum visits

        # Sort children by visits
        sorted_children = sorted(self.root_node.children, key=lambda c: c.visits, reverse=True)
        best_visits = sorted_children[0].visits
        second_visits = sorted_children[1].visits if len(sorted_children) > 1 else 0

        # Stop if best move has 80% of visits
        if best_visits > 0.8 * total_visits:
            return True

        # Stop if best move has 3x more visits than second best
        if second_visits > 0 and best_visits > 3 * second_visits:
            return True

        return False

    def simulate(self, expansion_count: int = 1):
        """Perform `expansion_count` simulations. Inner loop is optimized."""
        for _ in range(expansion_count):
            node = self.root_node
            # descend while expanded
            while node.expanded and node.children:
                node = node.get_preferred_child(self.root_node)
            self.expand(node)

    def simulate_batched(self, expansion_count: int = 1, batch_size: int = 8):
        """
        Perform simulations with batched NN evaluation.

        This is much faster than simulate() because it:
        1. Collects multiple leaf nodes before evaluation
        2. Evaluates them all in a single NN forward pass
        3. Uses virtual loss to prevent redundant exploration
        4. Implements early stopping to save time on obvious moves

        Args:
            expansion_count: Number of simulations to run
            batch_size: Number of nodes to collect before batched evaluation
        """
        if self.batch_evaluator is None:
            # Fallback to sequential if no batch evaluator provided
            return self.simulate(expansion_count)

        i = 0
        while i < expansion_count:
            # Check for early stopping after each batch
            if self.should_stop_early():
                print(f"⚡ Early stopping after {i} simulations (obvious best move)")
                break

            # Collect a batch of leaf nodes
            leaf_batch = []

            # Collect up to batch_size leaves (or remaining simulations)
            batch_limit = min(batch_size, expansion_count - i)

            for _ in range(batch_limit):
                # Select a leaf node
                node = self.root_node
                while node.expanded and node.children:
                    node = node.get_preferred_child(self.root_node)

                # Apply virtual loss to prevent other simulations from selecting this path
                node.add_virtual_loss()

                # Generate children if not yet expanded
                if not node.children:
                    self.child_finder(node, self)

                # Add to batch for evaluation
                leaf_batch.append(node)

            # Batch evaluate all collected leaves
            if leaf_batch:
                try:
                    # Get boards from nodes
                    boards = [node.state for node in leaf_batch]

                    # Batch evaluate - returns list of values
                    values = self.batch_evaluator(boards)

                    # Validate result length
                    if len(values) != len(leaf_batch):
                        raise RuntimeError(
                            f"Batch evaluator returned {len(values)} values for {len(leaf_batch)} boards"
                        )

                    # Expand and backup each node with its value
                    for node, value in zip(leaf_batch, values):
                        # Remove virtual loss
                        node.remove_virtual_loss()

                        # Update the node with the evaluated value
                        if value is not None:
                            node.update_win_value(value)

                        # Mark as expanded (prevents redundant evaluation)
                        node.expanded = True

                except Exception as e:
                    # On error, remove virtual losses and fall back to sequential
                    print(f"⚠️ Batch evaluation error: {e}, falling back to sequential")
                    for node in leaf_batch:
                        node.remove_virtual_loss()
                        self.expand(node)

            i += batch_limit

    def expand(self, node: Node):
        """Call child_finder to populate children, then evaluate each child fast."""
        self.child_finder(node, self)

        for child in node.children:
            # node_evaluator can return numeric (immediate eval) or None
            child_win_value = self.node_evaluator(child, self)
            if child_win_value is not None:
                # iterative update
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                # perform a single random rollout iteratively (avoid recursion)
                self._random_rollout_iter(child)
                # clear rollout-created children to reduce memory (original behavior)
                child.children = []

        # Mark as expanded (even terminal nodes with no children)
        node.expanded = True

    def _random_rollout_iter(self, node: Node):
        """Iterative random rollout until evaluator returns a value. Avoid recursion."""
        cur = node
        while True:
            # generate children for cur
            self.child_finder(cur, self)
            if not cur.children:
                # terminal or no moves: try to evaluate directly
                v = self.node_evaluator(cur, self)
                if v is not None:
                    cur.update_win_value(v)
                return

            choice = random.choice(cur.children)
            # if evaluator can score chosen child directly, update and stop
            val = self.node_evaluator(choice, self)
            if val is not None:
                # update parent chain from chosen child (so wins propagate as before)
                choice.update_win_value(val)
                return
            # otherwise continue from chosen child (like descent)
            # trim children of cur to only chosen to reduce allocations (original logic)
            cur.children = [choice]
            cur = choice

# Load the trained neural network model
print("🤖 Loading chess model from HuggingFace...")
from .models.lc0_inference import LC0ModelLoader
import os

# Environment variable to select which model to use (configurable per deployment slot)
# Examples:
#   - latest_v2_128x6.pt (small/fast model for Slot 1)
#   - latest_v2_256x10.pt (large/strong model for Slot 2)
#   - latest_transformer_v2_256x6h8.pt (transformer model for Slot 3)
MODEL_FILE = os.getenv("CHESS_MODEL_FILE", "lc0_128x8_epoch1.pt")
print(f"📦 Selected model: {MODEL_FILE}")

model_loader = LC0ModelLoader(
    repo_id="steveandcow/chesshacks-lc0",  # HuggingFace repo with trained models
    model_file=MODEL_FILE,  # Configurable via environment variable
    device="cpu"  # Use CPU for deployment (or "cuda" if GPU available)
)

try:
    model_loader.load_model()
    print("✅ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    print("Falling back to random policy...")
    MODEL_LOADED = False

import math
import random
import time
import traceback

# Persistent position cache across games (manual implementation with FIFO eviction)
POSITION_CACHE = {}
CACHE_STATS = {"hits": 0, "misses": 0, "total_lookups": 0}

def get_cached_position_eval(fen: str) -> Optional[float]:
    """Get cached evaluation for a position."""
    CACHE_STATS["total_lookups"] += 1
    result = POSITION_CACHE.get(fen)
    if result is not None:
        CACHE_STATS["hits"] += 1
    else:
        CACHE_STATS["misses"] += 1
    return result

def cache_position_eval(fen: str, value: float):
    """Cache evaluation for a position with FIFO eviction."""
    # Evict oldest entries if at capacity
    while len(POSITION_CACHE) >= 10000:
        try:
            POSITION_CACHE.pop(next(iter(POSITION_CACHE)))
        except (StopIteration, RuntimeError):
            # Cache is empty or was modified during iteration
            break
    POSITION_CACHE[fen] = value

def print_cache_stats():
    """Print cache statistics for debugging."""
    total = CACHE_STATS["total_lookups"]
    if total > 0:
        hit_rate = 100 * CACHE_STATS["hits"] / total
        print(f"📊 Cache: {CACHE_STATS['hits']} hits / {total} lookups ({hit_rate:.1f}% hit rate)")
    else:
        print("📊 Cache: No lookups yet")


# child_finder: avoid deepcopy by using board.copy() + push and avoid re-iterating generators.
def child_finder(node, montecarlo):
    board = node.state
    try:
        legal_moves = list(board.generate_legal_moves())
    except Exception:
        legal_moves = list(board.legal_moves)

    if not legal_moves:
        node.children = []
        return

    children = []
    for mv in legal_moves:
        child_board = board.copy()
        child_board.push(mv)

        child = Node(child_board)
        children.append(child)

    node.children = []
    for c in children:
        node.add_child(c)


# node_evaluator: short-circuit terminal states, cache results, and use NN if available.
def node_evaluator(node, montecarlo):
    # Use a small cached attribute to avoid re-evaluating (works even if Node doesn't define slots)
    cached = getattr(node, "_cached_eval_value", None)
    if cached is not None:
        return cached

    # Terminal checks should be extremely cheap; check them first.
    if node.state.is_variant_win():
        setattr(node, "_cached_eval_value", math.inf)
        return math.inf
    if node.state.is_variant_loss():
        setattr(node, "_cached_eval_value", -math.inf)
        return -math.inf

    # Check persistent position cache
    fen = node.state.fen()
    cached_value = get_cached_position_eval(fen)
    if cached_value is not None:
        setattr(node, "_cached_eval_value", cached_value)
        return cached_value

    # If there's a loaded model, prefer it. Keep NN calls guarded and lightweight in error handling.
    if MODEL_LOADED:
        try:
            # If model_loader supports batching, prefer collecting nodes then batching outside.
            val = model_loader.evaluate_position(node.state)
            # ensure a numeric return (guard)
            if val is None:
                # fallthrough to fallback heuristic
                pass
            else:
                # cache and return
                float_val = float(val)
                setattr(node, "_cached_eval_value", float_val)
                cache_position_eval(fen, float_val)  # Add to persistent cache
                return float_val
        except Exception as e:
            # Don't spam with stack traces inside hot loops — log once and fallback.
            # If you want persistent diagnostics, consider toggling a debug flag.
            print("⚠️ NN eval error; falling back to heuristic.")
            # Optionally log the exception to a file or a single-time reporter instead:
            # logger.exception("model eval failure")  # preferred in production

    # Fallback: use the node's internal get_score heuristic (fast)
    val = node.get_score(montecarlo.root_node)
    float_val = float(val)
    setattr(node, "_cached_eval_value", float_val)
    cache_position_eval(fen, float_val)  # Add to persistent cache
    return float_val


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Called every time the model needs to move.
    print("Cooking move...")

    # Create a cheap list of legal moves once.
    try:
        legal_moves = list(ctx.board.generate_legal_moves())
    except Exception:
        legal_moves = list(ctx.board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # If we have an NN, ask for policy/value.
    if MODEL_LOADED:
        try:
            move_probs, position_value = model_loader.predict(ctx.board)
            print(f"Position value: {position_value:.3f}")
            ctx.logProbabilities(move_probs)
        except Exception as e:
            print("⚠️ NN prediction failed; falling back to alternatives.")
            traceback.print_exc()
            move_probs = None

    # FAST OPENING MOVES (skip MCTS in opening to save time)
    move_number = ctx.board.fullmove_number
    if move_number <= 3:  # Opening phase (8 moves)
        print("⚡ OPENING: Using fast NN policy (no MCTS)")
        if MODEL_LOADED and move_probs:
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
            return best_move
        # Fallback to random if no model
        return random.choice(legal_moves)

    # ADAPTIVE TIME MANAGEMENT (improved)
    time_left_ms = getattr(ctx, "timeLeft", None)
    if time_left_ms is None:
        # If timeLeft not provided, default to conservative behavior (run some sims but limited)
        time_left_ms = 2000

    print(f"⏱️  Time remaining: {time_left_ms}ms")

    # With batch inference, we're much faster - adjust time estimates
    OVERHEAD_MS = 200   # lower overhead with batching
    # Batch inference is ~5-8x faster, so we can afford more simulations per ms
    MS_PER_SIMULATION = 15  # Much faster with batching (was 90)
    available_time = max(time_left_ms - OVERHEAD_MS, 0)
    max_simulations = int(available_time // MS_PER_SIMULATION)

    MIN_TIME_FOR_MCTS = 500
    MIN_SIMULATIONS = 4
    MAX_SIMULATIONS = 100  # Increased from 32 since we're faster with batching

    # If not enough time for meaningful MCTS, just use NN policy (or uniform if not available).
    if time_left_ms < MIN_TIME_FOR_MCTS or max_simulations < MIN_SIMULATIONS:
        print("⚡ FAST MODE: Using NN policy or fallback")
        if MODEL_LOADED and move_probs:
            # move_probs is expected as {move: prob} where move equals python-chess Move
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
            return best_move
        # fallback random weighted
        move_weights = [random.random() for _ in legal_moves]
        total = sum(move_weights)
        normalized = [w / total for w in move_weights]
        ctx.logProbabilities({m: p for m, p in zip(legal_moves, normalized)})
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # DYNAMIC SIMULATION COUNT based on position complexity
    num_legal_moves = len(legal_moves)
    if num_legal_moves < 5:  # Forced/obvious position
        base_simulations = min(20, max_simulations)
        print(f"🎯 SIMPLE POSITION: {num_legal_moves} legal moves")
    elif num_legal_moves > 30:  # Complex position
        base_simulations = min(MAX_SIMULATIONS, max_simulations)
        print(f"🎯 COMPLEX POSITION: {num_legal_moves} legal moves")
    else:
        base_simulations = min(50, max_simulations)
        print(f"🎯 NORMAL POSITION: {num_legal_moves} legal moves")

    # ADAPTIVE SIMULATION BUDGET based on policy entropy (NN confidence)
    # If NN is confident, use fewer simulations; if uncertain, use more
    if MODEL_LOADED and move_probs:
        # Calculate entropy of policy distribution (handle edge cases)
        if not move_probs:
            normalized_entropy = 0.5  # Default to medium confidence
        else:
            # Only sum over non-zero probabilities to avoid log(0)
            entropy = -sum(p * math.log(p + 1e-10) for p in move_probs.values() if p > 0)
            max_entropy = math.log(max(1, len(move_probs)))  # Avoid log(0)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Adjust simulations based on entropy
        if normalized_entropy < 0.3:  # Very confident NN
            entropy_multiplier = 0.5
            confidence_str = "CONFIDENT"
        elif normalized_entropy > 0.7:  # Very uncertain NN
            entropy_multiplier = 1.5
            confidence_str = "UNCERTAIN"
        else:
            entropy_multiplier = 1.0
            confidence_str = "NORMAL"

        num_simulations = int(base_simulations * entropy_multiplier)
        num_simulations = min(num_simulations, max_simulations)
        print(f"📊 NN {confidence_str} (entropy: {normalized_entropy:.2f}), simulations: {base_simulations} → {num_simulations}")
    else:
        num_simulations = base_simulations

    print(f"🎯 SEARCH MODE: Running {num_simulations} MCTS simulations")

    # Build a fen->move map efficiently (single copy + push/pop per move)
    fen_to_move = {}
    board = ctx.board
    for mv in legal_moves:
        board.push(mv)
        try:
            fen_to_move[board.fen()] = mv
        finally:
            board.pop()

    # Run MCTS with batch inference
    root = Node(board.copy())  # use copy so MonteCarlo mutates a separate tree
    montecarlo = MonteCarlo(root)
    montecarlo.child_finder = child_finder
    montecarlo.node_evaluator = node_evaluator

    # Set up batch evaluator if model is loaded
    if MODEL_LOADED:
        def batch_evaluator(boards):
            """Evaluate multiple boards at once using batched NN inference."""
            try:
                # Validate inputs
                if not boards:
                    print("⚠️ Empty board list received")
                    return []

                if not all(hasattr(b, 'turn') for b in boards):
                    print(f"⚠️ Invalid boards in batch: {boards}")
                    raise ValueError("Invalid board objects in batch")

                # Filter out terminal positions (shouldn't happen but be defensive)
                non_terminal_boards = []
                terminal_values = []
                board_indices = []

                for i, board in enumerate(boards):
                    if board.is_game_over():
                        # Terminal position - assign value based on outcome
                        result = board.result()
                        if result == "1-0":
                            terminal_values.append(1.0 if board.turn == chess.WHITE else -1.0)
                        elif result == "0-1":
                            terminal_values.append(-1.0 if board.turn == chess.WHITE else 1.0)
                        else:
                            terminal_values.append(0.0)  # Draw
                        board_indices.append(-1)  # Mark as terminal
                    else:
                        board_indices.append(len(non_terminal_boards))
                        non_terminal_boards.append(board)

                # Evaluate non-terminal positions
                if non_terminal_boards:
                    print(f"🔍 Batch evaluating {len(non_terminal_boards)} positions ({len(boards) - len(non_terminal_boards)} terminal)")
                    _, nn_values = model_loader.batch_predict(non_terminal_boards)
                    print(f"✓ Batch evaluation complete: {len(nn_values)} values returned")
                else:
                    print(f"⚠️ All {len(boards)} positions are terminal, no NN evaluation needed")
                    nn_values = []

                # Reconstruct full values list
                values = []
                terminal_idx = 0
                nn_idx = 0
                for idx in board_indices:
                    if idx == -1:
                        values.append(terminal_values[terminal_idx])
                        terminal_idx += 1
                    else:
                        values.append(nn_values[nn_idx])
                        nn_idx += 1

                return values
            except Exception as e:
                print(f"⚠️ Batch predict failed: {e}")
                traceback.print_exc()  # Print full stack trace
                print(f"⚠️ Falling back to sequential evaluation")
                try:
                    # Fallback to sequential evaluation
                    values = []
                    for i, b in enumerate(boards):
                        print(f"  Evaluating board {i+1}/{len(boards)}")
                        val = model_loader.evaluate_position(b)
                        values.append(val)
                    print(f"✓ Sequential evaluation complete: {len(values)} values")
                    return values
                except Exception as e2:
                    print(f"⚠️ Sequential fallback also failed: {e2}")
                    traceback.print_exc()
                    print(f"⚠️ Using neutral heuristic for all positions")
                    # Ultimate fallback: return 0.0 (neutral evaluation) for all positions
                    return [0.0] * len(boards)

        montecarlo.batch_evaluator = batch_evaluator

        # Use batched simulation for much faster search
        # Batch size of 8 is a good balance between GPU utilization and latency
        try:
            print(f"🚀 Starting batched MCTS with {num_simulations} simulations")
            montecarlo.simulate_batched(num_simulations, batch_size=8)
            print(f"✓ MCTS simulation complete")
        except Exception as e:
            print(f"⚠️ Batched MCTS failed: {e}")
            traceback.print_exc()
            print(f"⚠️ Falling back to sequential MCTS")
            # Clear any corrupted state
            montecarlo.batch_evaluator = None
            montecarlo.simulate(num_simulations)
    else:
        # Fallback to sequential if no model
        print(f"🚀 Starting sequential MCTS with {num_simulations} simulations")
        montecarlo.simulate(num_simulations)
        print(f"✓ MCTS simulation complete")

    try:
        best_child = montecarlo.make_choice()
    except Exception as e:
        print(f"⚠️ Failed to select best move from MCTS: {e}")
        traceback.print_exc()
        # Fallback to NN policy if available
        if MODEL_LOADED and move_probs:
            print("⚠️ Falling back to NN policy")
            return max(move_probs.items(), key=lambda x: x[1])[0]
        # Final fallback: random
        print("⚠️ Falling back to random move")
        return random.choice(legal_moves)

    # quick lookup: use fen map to get the original Move object without performing deep copies
    best_fen = best_child.state.fen()
    best_move = fen_to_move.get(best_fen)

    # Print cache statistics for monitoring performance
    print_cache_stats()

    if best_move:
        print(f"✓ MCTS selected move: {best_move.uci()}")
        return best_move

    # Fallback: use NN policy if available
    if MODEL_LOADED and move_probs:
        print("⚠️ MCTS selected unknown child; fallback to NN policy")
        return max(move_probs.items(), key=lambda x: x[1])[0]

    # Final fallback: random weighted
    print("⚠️ MCTS fallback to random policy")
    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    move_probs = {move: weight / total_weight for move, weight in zip(legal_moves, move_weights)}
    ctx.logProbabilities(move_probs)
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # Called when a new game starts. Clear caches on model and nodes if necessary.
    # If you cached any global values on nodes or the model, clear them here.
    # Example: clear a model internal cache, or reset a global transposition table if you add one.
    try:
        # If you have a transposition table
        # transposition_table.clear()
        # If model_loader has stateful caches:
        if MODEL_LOADED and hasattr(model_loader, "reset"):
            try:
                model_loader.reset()
            except Exception:
                pass
    except Exception:
        pass