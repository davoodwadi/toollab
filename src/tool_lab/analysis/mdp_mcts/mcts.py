import random
from dataclasses import dataclass
import numpy as np
import pandas as pd

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
        max_depth=None,
        seed=0,
        chance_enumeration_limit=500,
        tie_epsilon=1e-12,
        tie_break_reveal_penalty=0.0,
    ):
        self.products = list(products)
        self.attributes = list(attributes)

        self.n_products = len(self.products)
        self.n_attrs = len(self.attributes)
        self.num_bags = num_bags

        self.max_depth = max_depth
        self.rng = random.Random(seed)
        # np.random.seed(seed)

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

        return -(e_price * e_inv_w) * self.num_bags



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
    # Planner
    # -----------------------------

    def outcome_values(self, attr):
        # Initialize cache if it doesn't exist
        if not hasattr(self, "_outcomes_cache"):
            self._outcomes_cache = {}

        if attr.name not in self._outcomes_cache:
            if len(attr.domain) <= self.chance_enumeration_limit:
                self._outcomes_cache[attr.name] = attr.domain
            else:
                self._outcomes_cache[attr.name] = np.random.choice(
                    attr.domain, self.chance_enumeration_limit
                )

        return self._outcomes_cache[attr.name]


    def plan(self, state, depth=None):
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
            best_action = None

            # Option 1: choose one product: q is the value of choosing the best product NOW
            for product_i in range(self.n_products):
                q = self.choose_value(s, product_i)
                action = ("choose", product_i)

                if q > best_value:
                    best_value = q
                    best_action = action

            # Option 2: reveal one unknown attribute -> compute value for the resulting state
            if d > 0:
                for product_i in range(self.n_products):
                    for attr_i, attr in enumerate(self.attributes):
                        idx = self.index(product_i, attr_i)

                        if s[idx] is not None:
                            continue
                        
                        q = Q_reveal(s, product_i, attr_i, d)
                        # print(f'revealing {s}, {product_i}, {attr_i}: {q}')

                        if q > best_value:
                            best_value = q
                            best_action = ("reveal", product_i, attr_i)

            cache[key] = best_value, best_action
            return cache[key]

        def Q_reveal(s, product_i, attr_i, d):
            attr = self.attributes[attr_i]
            idx = self.index(product_i, attr_i)

            if s[idx] is not None:
                raise ValueError("Tried to evaluate reveal of known value.")

            outcomes = self.outcome_values(attr)

            total = 0.0

            for x in outcomes:
                # Fast tuple slice (no list allocation)
                new_state = s[:idx] + (x,) + s[idx+1:]
                
                child_value, _ = V(new_state, d - 1)
                total += child_value

            reveal_cost = attr.cost + self.tie_break_reveal_penalty
            return -reveal_cost + total / len(outcomes)


        value, best_action = V(state, depth)
        # Root action stats
        stats = []

        for action in self.legal_actions(state):
            if action[0] == "choose":
                _, product_i = action
                q = self.choose_value(state, product_i)
            else:
                if depth <= 0:
                    q = None
                else:
                    _, product_i, attr_i = action
                    q = Q_reveal(state, product_i, attr_i, depth)

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
            "best_action": best_action,
            "estimated_value": value,
            "stats": stats,
        }

    # -----------------------------
    # Rollout on one hidden realization
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
            print('+'*50)
            before = current_plan
            action = before["best_action"]

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
            step["next_action"] = self.format_action(after["best_action"])

            logs.append(step)

            state = new_state
            current_plan = after

        return logs



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

if __name__ == "__main__":
    products_leftdigit = ["coffee_a", "coffee_b"]
    products_discount = ["coffee_c", "coffee_d", "coffee_e"]

    TOOL_COST_LEFTDIGIT = 0.02
    TOOL_COST_DISCOUNT = 0.03
    # TOOL_COST = 10

    attributes_leftdigit = [
        Attribute(
            name="dollars",
            domain=np.linspace(1, 100, 50),
            cost=TOOL_COST_LEFTDIGIT,
        ),
        Attribute(
            name="weight",
            domain=np.linspace(1, 100, 50),
            cost=TOOL_COST_LEFTDIGIT,
        ),
        Attribute(
            name="cents", 
            domain=np.linspace(0, 99, 50).round(),
            cost=TOOL_COST_LEFTDIGIT,
        ),
    ]
    dollars = np.linspace(1, 100, 50)
    weights = np.linspace(10, 50, 50)
    d_e = dollars.mean()
    w_i_e = (1/weights).mean()
    value_e = d_e * w_i_e
    # print(value_e)
    # print((dollars).std())
    # print((1/weights).std())
    attributes_discount = [
        Attribute(
            name="dollars",
            domain=np.arange(4, 23),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="weight",
            domain=np.arange(4, 23),
            cost=TOOL_COST_DISCOUNT,
        ),
        Attribute(
            name="discount_percentage",
            domain=np.arange(0, 70, 5),
            cost=TOOL_COST_DISCOUNT,
        ),
    ]
    # print(attributes[1].domain)

    realization_leftdigit = {
        ("coffee_a", "dollars"): 4,
        ("coffee_a", "weight"): 10,
        ("coffee_a", "cents"): 99,

        ("coffee_b", "dollars"): 5,
        ("coffee_b", "weight"): 11,
        ("coffee_b", "cents"): 0,
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

    # realization = realization_leftdigit
    # products = products_leftdigit
    # attributes = attributes_leftdigit

    realization = realization_discount
    products = products_discount
    attributes = attributes_discount

    planner = ToolPlanner(
        products=products,
        attributes=attributes,
        num_bags=5,

        max_depth=3,
        # Exact enumeration for dollars and cents domains.
        chance_enumeration_limit=500,
        seed=123,

        # Keep 0.0 if tool cost is truly zero.
        # Set e.g. 1e-6 if you want to break ties against unnecessary reveals.
        tie_break_reveal_penalty=1e-6,
    )
    
    initial_state = planner.initial_state()
    # print(planner.pretty_state(initial_state))
    # for att in attributes:
    #     avg = att.domain.mean()
    #     print(f"attr: {att.name}, avg: {avg}")
    # print(planner.expected_price(initial_state, 0))
    # print(planner.expected_price(initial_state, 1))
    # legal_actions = planner.legal_actions(initial_state)
    # print(planner.format_action(legal_actions[2]))
    # for att in attributes:
    #     print(att.name, planner.outcome_values(att))


    











    logs = planner.rollout_verbose(realization)

    for i, step in enumerate(logs):
        print()
        print(f"Step {step['t']}")
        if i==(len(logs)-1):
            print("Prior state:", step["prior_state"])
        print("Estimated V(prior):", step["estimated_V_prior"])

        # print("Top action estimates:")
        # for s in step["top_action_estimates"]:
        #     print("   ", s["action"], "mean_q=", s["mean_q"])

        print("Action:", step["action"])

        if step["action"][0] == "reveal":
            pass
            # print("Observation:", step["observation"])
            # print("Posterior state:", step["posterior_state"])
            # print("Estimated V(posterior):", step["estimated_V_posterior"])
            # print("Next action:", step["next_action"])
        else:
            print("Chosen product:", step["chosen_product"])
