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

    @cache
    def V(self, state):
        best_value, best_action = self.stop_value(state)

        # Try every possible reveal
        for product_i in range(self.n):
            for attr_i, attr in enumerate(self.attributes):
                idx = self.index(product_i, attr_i)

                if state[idx] is not None:
                    continue

                expected_child_value = 0.0

                for x in attr.domain:
                    new_state = list(state)
                    new_state[idx] = x
                    new_state = tuple(new_state)

                    child_value, _ = self.V(new_state)
                    expected_child_value += child_value

                expected_child_value /= len(attr.domain)

                reveal_value = -attr.cost + expected_child_value

                # Strict improvement only.
                # This avoids revealing useless free information.
                if reveal_value > best_value + 1e-12:
                    best_value = reveal_value
                    best_action = (
                        "reveal",
                        self.products[product_i],
                        attr.name,
                    )

        return best_value, best_action

    def initial_state(self):
        return tuple([None] * (self.n * self.m))

    def rollout(self, realization, max_steps=100):
        """
        realization example:
        {
            ("A", "dollars"): 12,
            ("A", "cents"): 50,
            ("B", "dollars"): 10,
            ("B", "cents"): 99,
        }
        """

        state = self.initial_state()
        steps = []

        for _ in range(max_steps):
            value, action = self.V(state)

            if action[0] == "choose":
                steps.append({
                    "action": action,
                    "state_value": value,
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
            })

            state = new_state

        return steps


products = ["A", "B", "C"]

TOOL_COST = 0.

attributes = [
    Attribute("dollars", tuple(range(8, 23)), weight=100, cost=TOOL_COST),
    Attribute("cents", tuple(range(100)), weight=1, cost=TOOL_COST),
]

mdp = ExactPriceMDP(products, attributes)

realization = {
    ("A", "dollars"): 12,
    ("A", "cents"): 50,
    ("B", "dollars"): 12,
    ("B", "cents"): 20,
    ("C", "dollars"): 15,
    ("C", "cents"): 10,
}

steps = mdp.rollout(realization)

for step in steps:
    print(step)
