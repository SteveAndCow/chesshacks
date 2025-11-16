import math
from copy import deepcopy

import numpy as np

from .utils import chess_manager, GameContext
from chess import Move
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

        # avoid zero division
        visits_safe = visits if visits > 0 else 1
        parent_visits_safe = parent_visits if parent_visits > 0 else 1

        discovery_factor = self.discovery_factor
        policy = self.policy_value if (self.policy_value is not None) else 1.0

        # localize math functions
        log_parent = math.log(parent_visits_safe)
        sqrt_term = math.sqrt(log_parent / visits_safe)
        discovery_operand = discovery_factor * policy * sqrt_term

        # win operand uses sign depending on ownership relative to root player
        win_multiplier = 1.0 if parent.player_number == root_node.player_number else -1.0
        win_operand = win_multiplier * (self.win_value / visits_safe)

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


class MonteCarlo:
    __slots__ = ("root_node", "child_finder", "node_evaluator")

    def __init__(self, root_node: Node):
        self.root_node = root_node
        # Set these externally - child_finder(node, montecarlo) populates node.children
        self.child_finder: Callable[[Node, "MonteCarlo"], None] = lambda node, mc: None
        # node_evaluator(child, mc) -> optional numeric score (win_value) or None
        self.node_evaluator: Callable[[Node, "MonteCarlo"], Optional[float]] = lambda child, mc: None

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

    def simulate(self, expansion_count: int = 1):
        """Perform `expansion_count` simulations. Inner loop is optimized."""
        for _ in range(expansion_count):
            node = self.root_node
            # descend while expanded
            while node.expanded and node.children:
                node = node.get_preferred_child(self.root_node)
            self.expand(node)

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

        if node.children:
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
from .models.inference import ChessModelLoader

model_loader = ChessModelLoader(
    repo_id="steveandcow/chesshacks-bot",
    model_name="cnn_baseline",
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
                setattr(node, "_cached_eval_value", float(val))
                return float(val)
        except Exception as e:
            # Don't spam with stack traces inside hot loops — log once and fallback.
            # If you want persistent diagnostics, consider toggling a debug flag.
            print("⚠️ NN eval error; falling back to heuristic.")
            # Optionally log the exception to a file or a single-time reporter instead:
            # logger.exception("model eval failure")  # preferred in production

    # Fallback: use the node's internal get_score heuristic (fast)
    val = node.get_score(montecarlo.root_node)
    setattr(node, "_cached_eval_value", float(val))
    return float(val)


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

    # ADAPTIVE TIME MANAGEMENT (improved)
    time_left_ms = getattr(ctx, "timeLeft", None)
    if time_left_ms is None:
        # If timeLeft not provided, default to conservative behavior (run some sims but limited)
        time_left_ms = 2000

    print(f"⏱️  Time remaining: {time_left_ms}ms")

    OVERHEAD_MS = 250   # lower overhead assumed if we optimized NN batching etc.
    # If you can measure time per sim, replace MS_PER_SIMULATION with a dynamic estimate.
    MS_PER_SIMULATION = 90
    available_time = max(time_left_ms - OVERHEAD_MS, 0)
    max_simulations = int(available_time // MS_PER_SIMULATION)

    MIN_TIME_FOR_MCTS = 700
    MIN_SIMULATIONS = 2
    MAX_SIMULATIONS = 32

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

    # SEARCH MODE
    num_simulations = min(max_simulations, MAX_SIMULATIONS)
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

    # Run MCTS
    root = Node(board.copy())  # use copy so MonteCarlo mutates a separate tree
    montecarlo = MonteCarlo(root)
    montecarlo.child_finder = child_finder
    montecarlo.node_evaluator = node_evaluator

    # If your node_evaluator can be batched, consider running a micro-batching wrapper here.
    montecarlo.simulate(num_simulations)

    best_child = montecarlo.make_choice()

    # quick lookup: use fen map to get the original Move object without performing deep copies
    best_fen = best_child.state.fen()
    best_move = fen_to_move.get(best_fen)

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