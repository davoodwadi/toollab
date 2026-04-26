from dataclasses import dataclass
from functools import cache
import math


@dataclass(frozen=True)
class Attribute:
    name: str
    domain: tuple
    weight: float
    cost: float = 0.0


class ExactPriceMDP:
    def __init__(self, products, attributes):
        self.products = list(products)
        self.attributes = list(attributes)
        self.n = len(products)
        self.m = len(attributes)

        self.means = [
            sum(attr.domain) / len(attr.domain)
            for attr in self.attributes
        ]

    def index(self, product_i, attr_i):
        return product_i * self.m + attr_i

    def expected_price(self, state, product_i):
        total = 0.0

        for attr_i, attr in enumerate(self.attributes):
            idx = self.index(product_i, attr_i)
            value = state[idx]

            if value is None:
                value = self.means[attr_i]

            total += attr.weight * value

        return total

    def stop_value(self, state):
        prices = [
            self.expected_price(state, i)
            for i in range(self.n)
        ]

        best_i = min(range(self.n), key=lambda i: prices[i])
        best_price = prices[best_i]

        # utility = negative price
        return -best_price, ("choose", self.products[best_i])

    def better(self, cand_u, cand_steps, best_u, best_steps, eps=1e-12):
        if cand_u > best_u + eps:
            return True

        if abs(cand_u - best_u) <= eps and cand_steps < best_steps - eps:
            return True

        return False

    @cache
    def V(self, state):
        stop_u, stop_action = self.stop_value(state)

        best_u = stop_u
        best_steps = 0.0
        best_action = stop_action

        for product_i in range(self.n):
            for attr_i, attr in enumerate(self.attributes):
                idx = self.index(product_i, attr_i)

                if state[idx] is not None:
                    continue

                child_u_total = 0.0
                child_steps_total = 0.0

                for x in attr.domain:
                    new_state = list(state)
                    new_state[idx] = x
                    new_state = tuple(new_state)

                    child_u, child_steps, _ = self.V(new_state)

                    child_u_total += child_u
                    child_steps_total += child_steps

                reveal_u = -attr.cost + child_u_total / len(attr.domain)
                reveal_steps = 1.0 + child_steps_total / len(attr.domain)

                if self.better(reveal_u, reveal_steps, best_u, best_steps):
                    best_u = reveal_u
                    best_steps = reveal_steps
                    best_action = (
                        "reveal",
                        self.products[product_i],
                        attr.name,
                    )

        return best_u, best_steps, best_action


    def initial_state(self):
        return tuple([None] * (self.n * self.m))

    def rollout(self, realization, max_steps=100):
        state = self.initial_state()
        steps = []

        for _ in range(max_steps):
            value, expected_steps, action = self.V(state)

            if action[0] == "choose":
                steps.append({
                    "action": action,
                    "state_value": value,
                    "expected_remaining_reveals": expected_steps,
                })
                break

            _, product, attr_name = action
            observed_value = realization[(product, attr_name)]

            product_i = self.products.index(product)
            attr_i = [a.name for a in self.attributes].index(attr_name)
            idx = self.index(product_i, attr_i)

            new_state = list(state)
            new_state[idx] = observed_value
            new_state = tuple(new_state)

            steps.append({
                "action": action,
                "observed": observed_value,
                "state_value": value,
                "expected_remaining_reveals": expected_steps,
            })

            state = new_state

        return steps


    def rollout_verbose(self, realization, max_steps=100):
        state = self.initial_state()
        steps = []

        for t in range(max_steps):
            value_before, expected_steps_before, action = self.V(state)

            step = {
                "t": t,
                "prior_state": self.pretty_state(state),
                "value_before_action": value_before,
                "expected_remaining_reveals_before_action": expected_steps_before,
                "action": action,
            }

            if action[0] == "choose":
                step["chosen_product"] = action[1]
                steps.append(step)
                break

            _, product, attr_name = action
            observed_value = realization[(product, attr_name)]

            product_i = self.products.index(product)
            attr_i = [a.name for a in self.attributes].index(attr_name)
            idx = self.index(product_i, attr_i)

            new_state = list(state)
            new_state[idx] = observed_value
            new_state = tuple(new_state)

            value_after, expected_steps_after, next_action = self.V(new_state)

            step["observation"] = {
                "product": product,
                "attribute": attr_name,
                "value": observed_value,
            }
            step["posterior_state"] = self.pretty_state(new_state)
            step["value_after_observation"] = value_after
            step["expected_remaining_reveals_after_observation"] = expected_steps_after
            step["next_action_after_observation"] = next_action

            steps.append(step)

            state = new_state

        return steps

    def pretty_state(self, state):
        obs = {}

        for product_i, product in enumerate(self.products):
            obs[product] = {}

            for attr_i, attr in enumerate(self.attributes):
                idx = self.index(product_i, attr_i)
                value = state[idx]

                if value is not None:
                    obs[product][attr.name] = value

        return obs



products = ["A", "B"]

TOOL_COST = 0.

attributes = [
    Attribute("dollars", tuple(range(8, 23)), weight=100, cost=TOOL_COST),
    Attribute("cents", tuple(range(100)), weight=1, cost=TOOL_COST),
]

mdp = ExactPriceMDP(products, attributes)

realization = {
    ("A", "dollars"): 12,
    ("A", "cents"): 98,
    ("B", "dollars"): 12,
    ("B", "cents"): 99,
}

steps = mdp.rollout_verbose(realization)

for step in steps:
    print("+"*50)
    # print(step)
    print("+"*50)
    print(f"Step {step['t']}")
    print("Prior state:", step["prior_state"])
    print("Value before action:", step["value_before_action"])
    print("Expected remaining reveals before action:",
          step["expected_remaining_reveals_before_action"])
    print("Action:", step["action"])

    if step["action"][0] == "reveal":
        print("Observation:", step["observation"])
        print("Posterior state:", step["posterior_state"])
        print("Value after observation:", step["value_after_observation"])
        print("Next action after observation:", step["next_action_after_observation"])
    else:
        print("Chosen product:", step["chosen_product"])

