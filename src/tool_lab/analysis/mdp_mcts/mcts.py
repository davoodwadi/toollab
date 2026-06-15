import random
import math
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import beta

# -----------------------------
# Problem definition
# -----------------------------

@dataclass(frozen=True)
class Attribute:
    name: str
    domain: tuple
    cost: float = 0.0

# -----------------------------
# Sparse expectimax planner
# -----------------------------

class ToolPlanner:
    def __init__(
        self,
        products,
        attributes,
        num_bags=5,
        algorithm="expectimax",   
        mcts_iterations=50_000,     
        max_depth=None,
        chance_enumeration_limit=500,
        tie_epsilon=1e-12,
        tie_break_reveal_penalty=0.0,
    ):
        self.products = list(products)
        self.attributes = list(attributes)

        self.n_products = len(self.products)
        self.n_attrs = len(self.attributes)
        self.num_bags = num_bags

        self.algorithm = algorithm
        self.mcts_iterations = mcts_iterations

        self.max_depth = max_depth
        self.rng = random.Random()

        self.chance_enumeration_limit = chance_enumeration_limit

        self.tie_epsilon = tie_epsilon

        # Optional small planning-only penalty.
        # If > 0, ties prefer fewer reveals.
        self.tie_break_reveal_penalty = tie_break_reveal_penalty

        # --- Precompute values for speedup ---
        self.attr_indices = {attr.name: i for i, attr in enumerate(self.attributes)}
        self.precomputed_means = {}
        self.precomputed_inv_means = {}
        
        for i, attr in enumerate(self.attributes):
            self.precomputed_means[attr.name] = attr.domain.mean()
            
            # Create aliases for fast lookup in choose_value
            if attr.name in ["dollars", "price_dollars", "list_price"]:
                self.attr_indices["dollars"] = i
                self.precomputed_means["dollars"] = self.precomputed_means[attr.name]
                
            elif attr.name in ["cents", "price_cents"]:
                self.attr_indices["cents"] = i
                self.precomputed_means["cents"] = self.precomputed_means[attr.name]
                
            elif attr.name in ["discount", "discount_percentage"]:
                self.attr_indices["discount_percentage"] = i
                self.precomputed_means["discount_percentage"] = self.precomputed_means[attr.name]

            if attr.name in ["weight", 'weight_oz']:
                safe_domain = attr.domain[attr.domain != 0]
                self.precomputed_inv_means["weight"] = (1.0 / safe_domain).mean()

    # -----------------------------
    # State utilities
    # -----------------------------

    def index(self, product_i, attr_i):
        return product_i * self.n_attrs + attr_i

    def initial_state(self):
        return tuple([None] * (self.n_products * self.n_attrs))

    def remaining_reveals(self, state):
        return sum(x is None for x in state)

    def validate_state(self, state):
        expected_len = self.n_products * self.n_attrs
        if len(state) != expected_len:
            raise ValueError(
                f"Bad state length {len(state)}; expected {expected_len}"
            )

    def pretty_state(self, state):
        result = {}

        for i, product in enumerate(self.products):
            result[product] = {}
            for j, attr in enumerate(self.attributes):
                idx = self.index(i, j)
                value = state[idx]
                if value is not None:
                    result[product][attr.name] = value
        return result

    def current_tool_cost(self, state):
        cost = 0.0
        for i in range(self.n_products):
            for j, attr in enumerate(self.attributes):
                idx = self.index(i, j)
                if state[idx] is not None:
                    cost += attr.cost
        return cost

    # -----------------------------
    # Values
    # -----------------------------
    def choose_value(self, state, product_i):
        # 1. Base Price (Dollars)
        idx_d = self.index(product_i, self.attr_indices["dollars"])
        val_d = state[idx_d]
        e_price = float(val_d) if val_d is not None else self.precomputed_means["dollars"]

        # Cents
        if "cents" in self.attr_indices:
            idx_c = self.index(product_i, self.attr_indices["cents"])
            val_c = state[idx_c]
            e_price += 0.01 * (float(val_c) if val_c is not None else self.precomputed_means["cents"])

        # 2. discount_percentage
        if "discount_percentage" in self.attr_indices:
            idx_disc = self.index(product_i, self.attr_indices["discount_percentage"])
            val_disc = state[idx_disc]
            e_disc = float(val_disc) if val_disc is not None else self.precomputed_means["discount_percentage"]
            e_price *= (1.0 - (e_disc / 100.0))

        # 3. Inverse Weight
        idx_w = self.index(product_i, self.attr_indices["weight"])
        val_w = state[idx_w]
        e_inv_w = (1.0 / float(val_w)) if val_w is not None else self.precomputed_inv_means["weight"]

        # 4. Total Tool Cost incurred to reach this state
        t_cost = self.current_tool_cost(state)

        # EXACT MATH: ( E[Price] + ToolCost / num_bags ) * E[1 / Weight]
        expected_cost_per_bag = e_price + (t_cost / self.num_bags)
        expected_ratio = expected_cost_per_bag * e_inv_w

        return -expected_ratio # Negative because the planner maximizes


    def legal_actions(self, state):
        self.validate_state(state)

        actions = []
        # Reveal actions
        for i in range(self.n_products):
            for j in range(self.n_attrs):
                idx = self.index(i, j)
                if state[idx] is None:
                    actions.append(("reveal", i, j))
        # Choose actions
        for i in range(self.n_products):
            actions.append(("choose", i))
        return actions

    def format_action(self, action):
        if action[0] == "choose":
            _, product_i = action
            return ("choose", self.products[product_i])

        _, product_i, attr_i = action
        return (
            "reveal",
            self.products[product_i],
            self.attributes[attr_i].name,
        )

    # -----------------------------
    # Planner Sub-Engines
    # -----------------------------

    def outcome_values(self, attr):
        # uses self.chance_enumeration_limit to limit the domain of the attribute
        # Initialize cache if it doesn't exist
        if not hasattr(self, "_outcomes_cache"):
            self._outcomes_cache = {}

        if attr.name not in self._outcomes_cache:
            domain = attr.domain
            
            # If the domain is small, evaluate exactly
            if len(domain) <= self.chance_enumeration_limit:
                self._outcomes_cache[attr.name] = domain
            else:
                # Deterministic Quantization: Pick N evenly spaced indices
                # Example: for 100 items and limit 5, picks indices roughly at 10%, 30%, 50%, 70%, 90%
                limit = self.chance_enumeration_limit
                indices = np.linspace(0, len(domain) - 1, limit, dtype=int)
                self._outcomes_cache[attr.name] = domain[indices]
            # if attr.name=='dollars':
            #     print(attr)
            #     print('self._outcomes_cache[attr.name]', self._outcomes_cache[attr.name])
            #     print('attr.domain', attr.domain)

        return self._outcomes_cache[attr.name]


    def plan(self, state, depth=None):
        """Router function: calls the correct algorithm based on initialization."""
        if self.algorithm == "expectimax":
            return self._expectimax_plan(state, depth)
        elif self.algorithm == "mcts":
            return self._mcts_plan(state)
        elif self.algorithm == "myopic":
            return self._myopic_plan(state)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")



    def _expectimax_plan(self, state, depth=None):
        """
        Returns:
            {
                "best_action": raw action,
                "estimated_value": V(s),
                "stats": list of action Q-values
            }
        """
        self.validate_state(state)
        if depth is None:
            if self.max_depth is None:
                depth = self.remaining_reveals(state)
            else:
                depth = self.max_depth

        cache = {}

        def V(s, d):
            key = (s, d)
            if key in cache:
                return cache[key]

            best_value = float("-inf")
            best_actions = []

            # Option 1: choose one product: q is the value of choosing the best product NOW
            for product_i in range(self.n_products):
                q = self.choose_value(s, product_i)
                action = ("choose", product_i)

                if q > best_value + self.tie_epsilon:
                    best_value = q
                    best_actions = [action]
                elif abs(q - best_value) <= self.tie_epsilon:
                    best_actions.append(action)

            # Option 2: reveal one unknown attribute -> compute value for the resulting state
            if d > 0:
                for product_i in range(self.n_products):
                    for attr_i, attr in enumerate(self.attributes):
                        idx = self.index(product_i, attr_i)

                        if s[idx] is not None:
                            continue
                        
                        q = Q_reveal(s, product_i, attr_i, d)
                        # print(f'revealing {s}, {product_i}, {attr_i}: {q}')

                        if q > best_value + self.tie_epsilon:
                            best_value = q
                            best_actions = [("reveal", product_i, attr_i)]
                        elif abs(q - best_value) <= self.tie_epsilon:
                            best_actions.append(("reveal", product_i, attr_i))

            cache[key] = best_value, best_actions
            return cache[key]

        def Q_reveal(s, product_i, attr_i, d):
            attr = self.attributes[attr_i]
            idx = self.index(product_i, attr_i)
            outcomes = self.outcome_values(attr)
            total = 0.0

            for x in outcomes:
                # Fast tuple slice (no list allocation)
                new_state = s[:idx] + (x,) + s[idx+1:]                
                child_value, _ = V(new_state, d - 1)
                total += child_value

            average_value = total / len(outcomes)

            return -self.tie_break_reveal_penalty + average_value


        value, best_actions = V(state, depth)
        # Root action stats
        stats = []

        for action in self.legal_actions(state):
            if action[0] == "choose":
                q = self.choose_value(state, action[1])
            else:
                if depth <= 0:
                    q = None
                else:
                    _, p_idx, a_idx = action
                    q = Q_reveal(state, p_idx, a_idx, depth)

            stats.append({
                "action": self.format_action(action),
                "raw_action": action,
                "mean_q": q,
            })

        stats.sort(
            key=lambda s: float("-inf") if s["mean_q"] is None else s["mean_q"],
            reverse=True,
        )

        return {
            "best_actions": best_actions,
            "estimated_value": value,
            "stats": stats,
        }

    def _mcts_plan(self, state):
        self.validate_state(state)
        
        Q = defaultdict(float)
        N = defaultdict(int)
        N_s = defaultdict(int)
        V = {} # Stores the Bellman Max Value of a state

        # Initialize root value using the baseline greedy choice
        if state not in V:
            V[state] = max([self.choose_value(state, i) for i in range(self.n_products)])

        C_PARAM = 1.414 

        for _ in range(self.mcts_iterations):
            curr_state = state
            path = []
            
            # --- 1. SELECTION & EXPANSION ---
            while True:
                legal = self.legal_actions(curr_state)
                unexplored = [a for a in legal if N[(curr_state, a)] == 0]
                
                if unexplored:
                    # Expand one unexplored action
                    action = self.rng.choice(unexplored)
                    path.append((curr_state, action))
                    
                    # Generate the immediate outcome state
                    if action[0] == "choose":
                        leaf_state = curr_state
                    else:
                        _, p_i, a_i = action
                        val = self.rng.choice(self.attributes[a_i].domain)
                        idx = self.index(p_i, a_i)
                        leaf_state = curr_state[:idx] + (val,) + curr_state[idx+1:]
                    
                    # Initialize V for the new state
                    if leaf_state not in V:
                        V[leaf_state] = max([self.choose_value(leaf_state, i) for i in range(self.n_products)])
                        
                    break # Stop descending, we expanded a node
                    
                # All actions explored, use Local Normalized UCB
                best_ucb = float('-inf')
                best_action = None
                
                node_qs = [Q[(curr_state, a)] for a in legal]
                local_min = min(node_qs)
                local_max = max(node_qs)
                
                for a in legal:
                    q_val = Q[(curr_state, a)]
                    if local_max > local_min:
                        norm_q = (q_val - local_min) / (local_max - local_min)
                    else:
                        norm_q = 0.5
                        
                    ucb = norm_q + C_PARAM * math.sqrt(math.log(N_s[curr_state]) / N[(curr_state, a)])
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_action = a
                        
                action = best_action
                path.append((curr_state, action))
                
                if action[0] == "choose":
                    break
                    
                # Transition the chance node for the next selection loop
                _, p_i, a_i = action
                val = self.rng.choice(self.attributes[a_i].domain)
                idx = self.index(p_i, a_i)
                curr_state = curr_state[:idx] + (val,) + curr_state[idx+1:]
                
                if curr_state not in V:
                    V[curr_state] = max([self.choose_value(curr_state, i) for i in range(self.n_products)])

            # --- 2. EVALUATION & BELLMAN BACKPROPAGATION ---
            # No rollouts! We just walk backward up the path updating Exact Values.
            for i in reversed(range(len(path))):
                s, a = path[i]
                
                # The value flowing up is the exact V of the state resulting from action 'a'
                if a[0] == "choose":
                    step_reward = self.choose_value(s, a[1])
                else:
                    if i == len(path) - 1:
                        # This is the newly expanded leaf state
                        step_reward = V[leaf_state] - self.tie_break_reveal_penalty
                    else:
                        # Look at the state we actually transitioned to in the path
                        next_s = path[i+1][0]
                        step_reward = V[next_s] - self.tie_break_reveal_penalty
                
                # 1. Standard MCTS count updates
                N[(s, a)] += 1
                N_s[s] += 1
                
                # 2. Q(s, a) for a reveal action naturally becomes the EXACT Expected Value 
                #    (Average) of all the V(s') chance outcomes sampled!
                Q[(s, a)] += (step_reward - Q[(s, a)]) / N[(s, a)]
                
                # 3. BELLMAN MAX BACKUP: V(s) is the absolute MAX of its explored actions.
                legal_s = self.legal_actions(s)
                explored_qs = [Q[(s, act)] for act in legal_s if N[(s, act)] > 0]
                if explored_qs:
                    V[s] = max(explored_qs)

        # Compile results matching the dictionary format
        legal = self.legal_actions(state)
        # Pick the actions with the highest Expected Q-value
        best_value = max([Q[(state, a)] for a in legal])
        best_actions = [a for a in legal if abs(Q[(state, a)] - best_value) <= self.tie_epsilon]
        
        stats = []
        for a in legal:
            stats.append({
                "action": self.format_action(a),
                "raw_action": a,
                "mean_q": Q[(state, a)] if N[(state, a)] > 0 else None,
                "visits": N[(state, a)]
            })
            
        stats.sort(key=lambda st: float("-inf") if st["mean_q"] is None else st["mean_q"], reverse=True)
        
        return {"best_actions": best_actions, "estimated_value": best_value, "stats": stats}

    def _myopic_plan(self, state):
        """
        Calculates the Myopic Value of Information (EVSI).
        Assumes that after making ONE reveal, the agent MUST choose a product.
        """
        self.validate_state(state)
        
        q_values = {}
        best_value = float("-inf")
        best_actions = []

        # 1. Option A: Stop and Choose NOW (Current Value)
        for product_i in range(self.n_products):
            action = ("choose", product_i)
            q = self.choose_value(state, product_i)
            q_values[action] = q
            
            if q > best_value + self.tie_epsilon:
                best_value = q
                best_actions = [action]
            elif abs(q - best_value) <= self.tie_epsilon:
                best_actions.append(action)

        # 2. Option B: Reveal ONE attribute, then force a choice
        for product_i in range(self.n_products):
            for attr_i, attr in enumerate(self.attributes):
                idx = self.index(product_i, attr_i)

                # Skip if already revealed
                if state[idx] is not None:
                    continue
                    
                action = ("reveal", product_i, attr_i)
                outcomes = self.outcome_values(attr)
                total_expected_next_value = 0.0

                # Simulate every possible outcome of this tool call
                for x in outcomes:
                    next_state = state[:idx] + (x,) + state[idx+1:]
                    
                    # Since it's Myopic, we assume we MUST choose a product in the next state.
                    # We find the max value of choosing in that future state.
                    best_next_choose = max([
                        self.choose_value(next_state, next_p_i) 
                        for next_p_i in range(self.n_products)
                    ])
                    
                    total_expected_next_value += best_next_choose

                # Average the future values and apply the tie breaker
                avg_next_value = total_expected_next_value / len(outcomes)
                q = avg_next_value - self.tie_break_reveal_penalty
                
                q_values[action] = q
                
                if q > best_value + self.tie_epsilon:
                    best_value = q
                    best_actions = [action]
                elif abs(q - best_value) <= self.tie_epsilon:
                    best_actions.append(action)

        # 3. Format the stats exactly like expectimax/mcts for display
        stats = []
        for action, q in q_values.items():
            stats.append({
                "action": self.format_action(action),
                "raw_action": action,
                "mean_q": q,
            })

        stats.sort(
            key=lambda s: float("-inf") if s["mean_q"] is None else s["mean_q"],
            reverse=True,
        )

        return {
            "best_actions": best_actions,
            "estimated_value": best_value,
            "stats": stats,
        }

    # -----------------------------
    # Rollout and Display
    # -----------------------------

    def apply_real_observation(self, state, realization, action):
        if action[0] != "reveal":
            raise ValueError("Only reveal actions produce observations.")

        _, product_i, attr_i = action

        idx = self.index(product_i, attr_i)

        if state[idx] is not None:
            raise ValueError(
                f"Tried to reveal already observed value: "
                f"{self.products[product_i]} {self.attributes[attr_i].name}"
            )

        product = self.products[product_i]
        attr = self.attributes[attr_i]

        observed_value = realization[(product, attr.name)]

        new_state = list(state)
        new_state[idx] = observed_value

        return tuple(new_state), observed_value

    def rollout_verbose(self, realization, max_steps=100, show_top_k=5):
        state = self.initial_state()
        logs = []

        current_plan = self.plan(state)

        for t in range(max_steps):
            print('+'*50)
            print(f't: {t}')
            self.display_plan_evaluations(state, current_plan)
            print('+'*50)
            before = current_plan
            best_actions = before["best_actions"]
            
            if len(best_actions) > 1:
                formatted_actions = [self.format_action(a) for a in best_actions]
                print(f"Found {len(best_actions)} optimal actions with equal values: {formatted_actions}")
                action = self.rng.choice(best_actions)
                print(f"Randomly selecting {self.format_action(action)} for this rollout step.")
            else:
                action = best_actions[0]

            step = {
                "t": t,
                "prior_state": self.pretty_state(state),
                "estimated_V_prior": before["estimated_value"],
                "action": self.format_action(action),
                "top_action_estimates": before["stats"][:show_top_k],
            }

            if action[0] == "choose":
                _, product_i = action
                step["chosen_product"] = self.products[product_i]
                logs.append(step)
                break

            new_state, observed_value = self.apply_real_observation(
                state,
                realization,
                action,
            )

            # Plan from new_state.
            after = self.plan(new_state)

            step["observation"] = {
                "product": self.format_action(action)[1],
                "attribute": self.format_action(action)[2],
                "value": observed_value,
            }
            step["posterior_state"] = self.pretty_state(new_state)
            step["estimated_V_posterior"] = after["estimated_value"]
            step["next_actions"] = [self.format_action(a) for a in after["best_actions"]]

            logs.append(step)

            state = new_state
            current_plan = after

        return logs

    def display_plan_evaluations(self, state, plan_result):
        """Prints a human-readable leaderboard of actions and their math."""
        print(f"\n{'='*85}")
        print(f"CURRENT STATE: {self.pretty_state(state)}")
        
        # Calculate current tool cost based on what is revealed
        current_tc = self.current_tool_cost(state)
                    
        print(f"Accumulated Tool Cost: ${current_tc:.2f} (or ${current_tc/self.num_bags:.4f} per bag)")
        print(f"{'-'*85}")
        print(f"{'Rank':<5} | {'Action':<25} | {'Exp. Cost / Weight':<18} | {'Math Breakdown'}")
        print(f"{'-'*85}")
        
        # plan_result["stats"] is already sorted by highest Q-value (which is the lowest Cost)
        for rank, stat in enumerate(plan_result["stats"], 1):
            action = stat["raw_action"]
            q = stat["mean_q"]
            
            # Format action string nicely (e.g., "choose coffee_a" or "reveal coffee_b dollars")
            formatted = self.format_action(action)
            action_str = f"{formatted[0].upper()} {formatted[1]}"
            if len(formatted) == 3:
                action_str += f" {formatted[2]}"
                
            # Flip Q back to positive Expected Cost/Weight
            cost_weight_ratio = -q if q is not None else float('inf')
            
            breakdown = ""
            if action[0] == "choose":
                _, p_i = action
                
                # Re-calculate components just for the printout
                idx_d = self.index(p_i, self.attr_indices["dollars"])
                val_d = state[idx_d]
                e_price = float(val_d) if val_d is not None else self.precomputed_means["dollars"]
                
                if "cents" in self.attr_indices:
                    idx_c = self.index(p_i, self.attr_indices["cents"])
                    val_c = state[idx_c]
                    e_price += 0.01 * (float(val_c) if val_c is not None else self.precomputed_means["cents"])
                if "discount_percentage" in self.attr_indices:
                    idx_disc = self.index(p_i, self.attr_indices["discount_percentage"])
                    val_disc = state[idx_disc]
                    e_disc = float(val_disc) if val_disc is not None else self.precomputed_means["discount_percentage"]
                    e_price *= (1.0 - (e_disc / 100.0))

                idx_w = self.index(p_i, self.attr_indices["weight"])
                val_w = state[idx_w]
                e_inv_w = (1.0 / float(val_w)) if val_w is not None else self.precomputed_inv_means["weight"]
                
                t_cost_per_bag = current_tc / self.num_bags
                
                breakdown = f"(Price: ${e_price:.2f} + Tool: ${t_cost_per_bag:.4f}) * E[1/W]: {e_inv_w:.4f}"
            else:
                # print(action[2], self.outcome_values(self.attributes[action[2]]))
                outcomes = len(self.outcome_values(self.attributes[action[2]]))
                breakdown = f"Expectimax average of {outcomes} possible future states"
                
            print(f"{rank:<5} | {action_str:<25} | {cost_weight_ratio:<18.4f} | {breakdown}")
            
        print(f"{'='*85}\n")


def evaluate_human_trajectory(planner, realization, human_actions, trial_id, tool_cost):
    """
    Replays a human's sequence of actions through the optimal planner.
    
    human_actions: list of formatted actions, e.g., 
                   [('reveal', 'A', 'dollars'), ('reveal', 'B', 'weight'), ('choose', 'B')]
    """
    state = planner.initial_state()
    step_records = []
    
    for t, h_action in enumerate(human_actions):
        # 1. Ask the optimal planner to evaluate the CURRENT state
        plan = planner.plan(state)
        
        # 2. Extract Q-values for all legal actions
        # (Assuming your human_actions format matches your planner.format_action output)
        action_stats = {s["action"]: s["mean_q"] for s in plan["stats"]}
        
        max_q = max(action_stats.values())
        human_q = action_stats.get(h_action, None)
        
        # 3. Calculate metrics
        if human_q is None:
            raise ValueError(f"Human took illegal action: {h_action} at state {state}")
            
        regret = max_q - human_q
        is_optimal = regret <= 1e-6  # Accounts for floating point inaccuracies
        
        # 4. Record step data
        step_records.append({
            "trial_id": trial_id,
            "tool_cost": tool_cost,
            "step_t": t,
            "state": planner.pretty_state(state),
            "human_action": h_action,
            "human_q": human_q,
            "max_q": max_q,
            "regret": regret,
            "is_optimal": is_optimal,
            "is_reveal": h_action[0] == "reveal"
        })
        
        # 5. Transition to the next state EXACTLY as the human experienced it
        if h_action[0] == "reveal":
            # Find the raw action tuple (e.g., ("reveal", 0, 1)) to feed to apply_real_observation
            raw_action = next(s["raw_action"] for s in plan["stats"] if s["action"] == h_action)
            state, _ = planner.apply_real_observation(state, realization, raw_action)
            
        elif h_action[0] == "choose":
            break # Trial ends
            
    return pd.DataFrame(step_records)

# -----------------------------
# Example
# -----------------------------

def get_skewed_domain_int(lower, upper):

    # 1. Define the parameters for the right-skewed Beta distribution
    # a=2, b=8 creates a strong right-skew (peak on the left, long tail to the right)
    a_param, b_param = 2, 8

    # 2. Generate 50 equally spaced percentiles (from 1% to 99%)
    percentiles = np.linspace(0.01, 0.99, 50)

    # 3. Generate the right-skewed domain bounded between 0 and 1
    skewed_base = beta.ppf(percentiles, a_param, b_param)

    # 4. Scale the domain to your specific range [2, 35]
    range_val = upper - lower

    skewed_domain_continuous = lower + (skewed_base * range_val)

    # If this is for "dollars" and needs to be integers:
    skewed_domain_integers = np.round(skewed_domain_continuous).astype(int)

    return skewed_domain_integers

def get_price_dollars_instant_coffee_distribution_int(lower=2, upper=40):
    # 2. Simulate the continuous prices using a Log-Normal distribution
    # A mean of 2.0 and sigma of 0.5 in log-space creates a peak around $6-$8 
    # and a long tail stretching toward the $30s.
    mu = 2.0
    sigma = 0.5
    num_products = 100

    continuous_prices = np.random.lognormal(mean=mu, sigma=sigma, size=num_products)

    # 3. Apply realistic market bounds (minimum $2, hard cap at $40 for standard retail)
    continuous_prices = np.clip(continuous_prices, a_min=lower, a_max=upper)

    # 4. Extract the `price_dollars` (Whole dollar amount / integer floor)
    price_dollars = np.floor(continuous_prices).astype(int)
    return price_dollars

def get_price_dollars_instant_coffee_distribution_float(lower=2, upper=40):
    # 2. Simulate the continuous prices using a Log-Normal distribution
    # A mean of 2.0 and sigma of 0.5 in log-space creates a peak around $6-$8 
    # and a long tail stretching toward the $30s.
    mu = 2.0
    sigma = 0.5
    num_products = 100

    continuous_prices = np.random.lognormal(mean=mu, sigma=sigma, size=num_products)

    # 3. Apply realistic market bounds (minimum $2, hard cap at $40 for standard retail)
    continuous_prices = np.clip(continuous_prices, a_min=lower, a_max=upper)

    # 4. Extract the `price_dollars` (Whole dollar amount / integer floor)
    price_dollars = np.floor(continuous_prices).astype(float)
    return price_dollars


class InstantCoffeeWeightDistribution:
    def __init__(self):
        # Standard sizes in ounces
        self.weights_oz = [3.5, 7.0, 8.0, 12.0, 16.0]
        
        # Estimated probability mass for each size
        self.probabilities = [0.15, 0.35, 0.35, 0.10, 0.05]
        
        # Validate that probabilities sum to 1
        assert np.isclose(sum(self.probabilities), 1.0), "Probabilities must sum to 1"

    def sample_weights(self, num_samples=100):
        """
        Generates 'num_samples' of instant coffee weights based on the distribution.
        """
        return np.random.choice(
            self.weights_oz, 
            size=num_samples, 
            p=self.probabilities
        )


if __name__ == "__main__":
    products_leftdigit = ["coffee_a", "coffee_b"]
    products_discount = ["coffee_c", "coffee_d", "coffee_e"]
    products_discount2 = ["coffee_a", "coffee_b"]
    # random.shuffle(products_discount)
    TOOL_COST_LEFTDIGIT = 0.
    TOOL_COST_DISCOUNT = 0.001
    # TOOL_COST = 10

    attributes_leftdigit = [
        Attribute(
            name="dollars",
            domain=get_price_dollars_instant_coffee_distribution_int(2, 40),
            cost=TOOL_COST_LEFTDIGIT,
        ),
        Attribute(
            name="weight",
            domain=InstantCoffeeWeightDistribution().sample_weights(),
            cost=TOOL_COST_LEFTDIGIT,
        ),
        Attribute(
            name="cents", 
            domain=np.linspace(0, 99, 50).round(),
            cost=TOOL_COST_LEFTDIGIT,
        ),

        Attribute(name="origin", domain=np.array([0,1]), cost=TOOL_COST_LEFTDIGIT),
        Attribute(name="roast_date", domain=np.array([0,1]), cost=TOOL_COST_LEFTDIGIT),
        Attribute(name="packaging", domain=np.array([0,1]), cost=TOOL_COST_LEFTDIGIT),
        Attribute(name="calories", domain=np.arange(50), cost=TOOL_COST_LEFTDIGIT),
    ]

    attributes_discount = [
        Attribute(
            name="dollars",
            domain=np.linspace(8, 15, 50),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="weight",
            domain=np.linspace(8, 12, 50),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="discount_percentage",
            domain=np.array([0,20]),
            cost=TOOL_COST_DISCOUNT,
        ),
    ]

    attributes_discount2 = [
        Attribute(
            name="dollars",
            domain=np.linspace(12, 15, 50),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="weight",
            domain=np.linspace(10, 10.1, 50),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="discount_percentage",
            domain=np.array([0,20]),
            cost=TOOL_COST_DISCOUNT,
        ),
    ]

    # print(attributes[1].domain)

    realization_leftdigit = {
        ("coffee_a", "dollars"): 4,
        ("coffee_a", "cents"): 99,
        ("coffee_a", "weight"): 10,
        ("coffee_a", "roast_date"): 0,
        ("coffee_a", "origin"): 0,
        ("coffee_a", "packaging"): 0,
        ("coffee_a", "calories"): 3,

        ("coffee_b", "dollars"): 5,
        ("coffee_b", "cents"): 0,
        ("coffee_b", "weight"): 11,
        ("coffee_b", "roast_date"): 0,
        ("coffee_b", "origin"): 0,
        ("coffee_b", "packaging"): 0,
        ("coffee_b", "calories"): 3,
    }


    realization_discount = {
        ("coffee_c", "dollars"): 8,
        ("coffee_c", "weight"): 8,
        ("coffee_c", "discount_percentage"): 0,

        ("coffee_d", "dollars"): 15,
        ("coffee_d", "weight"): 12,
        ("coffee_d", "discount_percentage"): 20,

        # ("A", "cents"): 91,
        ("coffee_e", "dollars"): 10,
        ("coffee_e", "weight"): 10.9,
        ("coffee_e", "discount_percentage"): 0,
        # ("B", "cents"): 90,
    }


    realization_discount2 = {
        ("coffee_a", "dollars"): 15,
        ("coffee_a", "weight"): 10.,
        ("coffee_a", "discount_percentage"): 20,

        ("coffee_b", "dollars"): 12,
        ("coffee_b", "weight"): 10.1,
        ("coffee_b", "discount_percentage"): 0,
    }

    realization = realization_leftdigit
    products = products_leftdigit
    attributes = attributes_leftdigit

    # realization = realization_discount
    # products = products_discount
    # attributes = attributes_discount

    planner = ToolPlanner(
        products=products,
        attributes=attributes,
        num_bags=5,

        algorithm="expectimax",       # <--- SWITCH TO EXPECTIMAX
        max_depth=3,                  # <--- SET DEPTH TO 2 (or 3)
        chance_enumeration_limit=10,   # <--- BUCKETS THE 99 OUTCOMES INTO 5

        # algorithm="myopic",       
        # chance_enumeration_limit=5,   # <--- BUCKETS THE 99 OUTCOMES INTO 5


        # max_depth=None,
        # Exact enumeration for dollars and cents domains.
        # chance_enumeration_limit=5,

        # Keep 0.0 if tool cost is truly zero.
        # Set e.g. 1e-6 if you want to break ties against unnecessary reveals.
        tie_break_reveal_penalty=1e-6,
    )


    logs = planner.rollout_verbose(realization)

    for i, step in enumerate(logs):
        print()
        print(f"Step {step['t']}")
        if i==(len(logs)-1):
            print("Prior state:", step["prior_state"])
        print("Estimated V(prior):", round(step["estimated_V_prior"], 4))

        # print("Top action estimates:")
        # for s in step["top_action_estimates"]:
        #     print("   ", s["action"], "mean_q=", s["mean_q"])

        print("Action:", step["action"])

        if step["action"][0] == "reveal":
            pass
            # print("Observation:", step["observation"])
            # print("Posterior state:", step["posterior_state"])
            # print("Estimated V(posterior):", step["estimated_V_posterior"])
            # print("Next actions:", step["next_actions"])
        else:
            print('+'*50)
            print("Chosen product:", step["chosen_product"])
            print('+'*50)

    # print(products)

# for att in attributes:
#     expected = att.domain.mean()
#     print(f'expected {att.name}: {expected}')

# print(np.sort(get_price_dollars_instant_coffee_distribution(2, 40)))
# print(np.sort(InstantCoffeeWeightDistribution().sample_weights()))
