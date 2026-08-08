Tool-lab is our proposed method for analyzing llm behavior. It adapt the classic human process tracing method from cognitive psychology, that is mouselab mdp, to test llms in practical contexts to see whether the llms reasoning and information acquisition is optimal from a meta planning perspective. We want to write a paper introducing the method.

tldr: Tool-Lab presents pieces of information behind tool calls for LLMs. The LLM has to decide which piece of information to acquire, when to stop, and what final decision to make.

# Experiments

# C1: Mouselab-Style Reward Trees — Full Experimental Design


## 1. Environment Specification

### Structure
- **Directed tree/graph** with a start node; LLM must select a path from root to leaf
- **Rewards** are hidden at each node, revealed only via a tool call ("reveal node X")
- Each reveal incurs a **cost** (subtracted from final payoff)
- **Payoff** = sum of rewards along chosen path − total reveal cost

### Canonical Layout (from Callaway et al.)
- **3-step, 2-branch tree**: 12–13 nodes, depth 3
- Rewards drawn from discrete distributions (e.g., {−10, −5, 5, 10})
- Small enough for **exact optimal metalevel policy via backward induction**

### Tool Interface
```
Tools available:
  reveal(node_id) → returns hidden reward; costs c
  choose_path(path) → terminates episode, returns payoff
```
LLM sees the tree topology (which nodes exist, adjacency) but not the values until revealed.

---

## 2. Independent Variables (Manipulations)

| Variable | Levels | Purpose |
|----------|--------|---------|
| **Reward variance structure** | Increasing / Decreasing / Constant across depth | Core: tests strategy adaptation (E3) |
| **Reveal cost** | Low / Medium / High | Cost sensitivity (E2) |
| **Tree depth/branching** | Shallow (3×2) / Deep (5×2) | Scaling of planning |
| **Model** | GPT, Claude, Llama, Qwen, o1-style | Family/scale comparison (E6) |

### Why variance structure matters (the key manipulation)
- **Increasing variance** (big rewards deep): optimal policy is *far-sighted*—inspect leaves first
- **Decreasing variance** (big rewards early): optimal policy is *near-sighted*—inspect roots first
- **Constant**: intermediate
- Distinct optimal "inspection signatures" → you can classify the LLM's *strategy*, not just efficiency

---

## 3. Computing the Normative Optimum

- Formulate as a **metalevel MDP**: states = set of revealed nodes + values; actions = reveal another node OR stop-and-choose
- **Backward induction** for small trees gives exact optimal policy and its expected value
- For deeper trees, use **BMPS** (Callaway et al.) as tractable near-optimal approximation
- Precompute for each environment configuration → your ground-truth benchmark

**Baselines to report alongside optimum:**
- Random reveal + random choice
- Zero-reveal (choose blindly)
- Full-reveal (inspect everything, then choose) — the "no cost-awareness" ceiling
- Myopic Value of Information (greedy one-step VOI)

---

## 4. Dependent Variables (Metrics)

### Performance
- **Normalized payoff:** (LLM − random) / (optimal − random) ∈ interpretable scale
- **Metalevel regret:** optimal expected value − LLM expected value

### Process (the novel part)
- **Number of reveals** (vs. optimal count) → over/under-acquisition
- **Inspection order signature:** distribution over depth-of-first-reveals; compare to optimal far/near-sighted pattern
- **Stopping accuracy:** did it stop when marginal VOI < cost?
- **Strategy classification:** fit behavior to known Mouselab strategy set (random, myopic, far-sighted, near-sighted, optimal)

### Cost-sensitivity slope
- Regression of #reveals on cost level; compare slope to optimal slope

---

## 5. Protocol

1. Generate N environment instances per configuration (e.g., N=100), with fixed random seeds for reproducibility
2. For each, precompute optimal policy + baselines
3. Run each model with **repeated trials** (≥5) to estimate behavioral distribution and handle stochasticity (temperature > 0)
4. Log the **full tool-call trace** + reasoning tokens
5. Control prompt held constant across conditions; report prompt in appendix

### Prompt conditions (crossed or as ablation, links to E5)
- Minimal (just rules + tools)
- CoT ("think step by step")
- VOI-primed ("consider whether revealing is worth its cost")

---

## 6. Analysis Plan

- **H1 (suboptimality):** LLM normalized payoff < 1 (below optimal) — report effect size + CI
- **H2 (cost insensitivity):** LLM cost-slope flatter than optimal slope → mixed-effects model
- **H3 (strategy mismatch):** LLM inspection signatures don't shift with variance structure the way optimal does → χ² / distribution comparison
- **H4 (model differences):** reasoning models closer to optimal? — ANOVA/mixed model across families
- Mixed-effects models with random effects for environment instance; report significance + effect sizes

---

## 7. Key Figures for the Paper

1. **Bar plot:** normalized payoff, all models vs. baselines/optimum
2. **Line plot:** #reveals vs. cost, LLM curves against optimal curve (cost sensitivity)
3. **Strategy heatmap:** inspection-depth distribution per variance condition, LLM vs. optimal (the "signature" figure — very compelling)
4. **Confusion-style plot:** proportion of trials classified into each strategy type per model

---

## 8. Expected/Interesting Findings

- LLMs **over-reveal** (cost-insensitive), inflating cost with little payoff gain
- LLMs **fail to switch strategy** with variance structure (use one default heuristic)
- Reasoning models reveal *more* but not necessarily *smarter* — longer, not more rational
- Semantic-free abstract setting isolates pure meta-planning ability (contrast with C4 later)


---

# C2: Multi-Armed Bandit with Costly Probes — Experiment Design

This context has a clean, well-understood value-of-information structure, making the optimal metalevel policy tractable. Here's a complete design.

---

## 1. Task Definition

### Setup
- There are **K arms**, each with an unknown reward drawn from a known prior.
- The LLM can **probe** an arm (a tool call) to observe a noisy or exact sample of its reward, paying a **cost c** per probe.
- After any number of probes, the LLM **commits** to one arm and receives that arm's true reward.
- **Objective:** maximize `E[reward of chosen arm] − c × (number of probes)`.

### The meta-decision (what we're measuring)
At each step the LLM chooses: **probe arm i**, or **stop and commit to arm j**. This is a pure *metalevel* control problem—exactly the Mouselab logic, but with a bandit structure.

---

## 2. Formalizing the Optimal Policy

### Belief state
- Prior on each arm: e.g., `μ_i ~ Normal(m_i, σ_i²)`.
- Probing arm i returns a sample; belief updates via Bayesian conjugate update.
- **State = posterior parameters for all arms** `(m_i, σ_i²)_{i=1..K}`.

### Optimal metalevel MDP
- **Actions:** {probe_1, ..., probe_K, stop}.
- **Reward of stopping:** `max_i m_i` (best current expected value).
- **Cost of probing:** `c`, then transition to updated belief.
- **Solve via backward induction** on a bounded horizon (max probes = H), or value iteration on discretized belief state.

### Practical computability
- **Exact:** small K (2–4), discretized rewards, short horizon → exact backward induction.
- **Bernoulli arms** (Beta-Bernoulli): belief state is integer counts `(α_i, β_i)`, so the state space is naturally discrete and small → **exact optimum is easy**. *I recommend Bernoulli arms as the primary version.*

---

## 3. Independent Variables (Manipulations)

| Variable | Levels | Tests |
|----------|--------|-------|
| **Probe cost (c)** | low / medium / high | Cost sensitivity (E2) — # probes should decrease as c rises |
| **Number of arms (K)** | 2, 4, 8 | Scaling of search breadth |
| **Prior spread** | tight vs. wide priors | Should probe more when uncertain |
| **Prior asymmetry** | uniform vs. one clearly-best arm | Rational agent skips probing dominated arms |
| **Probe informativeness** | exact reveal vs. noisy sample | Value of repeated probing |

The **cost × prior-spread** interaction is your richest result: optimal behavior is highly structured there.

---

## 4. Dependent Variables (Metrics)

### Primary
1. **Net value achieved** vs. optimal-policy value → *optimality gap / regret*.
2. **Number of probes** vs. optimal number.

### Process-level (the novel part)
3. **Which arms probed** — does the LLM probe promising/uncertain arms (optimal) or spread uniformly / fixate?
4. **Stopping calibration** — probe-count as a function of belief entropy; compare slope to optimal.
5. **Cost-elasticity** — regression of #probes on c; compare LLM slope vs. optimal slope.
6. **Dominated-arm probing rate** — how often it wastes probes on clearly-bad arms.

### Bias taxonomy (for narrative)
- **Over-exploration:** probes > optimal at fixed cost.
- **Under-exploration / laziness:** commits too early.
- **Cost insensitivity:** flat #probes across c.
- **Uniform search:** ignores prior asymmetry.

---

## 5. Conditions / Baselines

- **Optimal metalevel policy** (upper bound).
- **Random probing + random stop** (lower bound).
- **Greedy/myopic VOI** (probe arm with highest one-step value of information; stop when no probe has positive myopic VOI) — a strong, interpretable baseline. *Comparing LLM to myopic-VOI reveals whether LLMs are even myopically rational.*
- **Full-information oracle** (reference for achievable reward).

---

## 6. Prompting Conditions (crossed with above)

- Zero-shot (just task + tool schema)
- CoT ("think step by step")
- **VOI-explicit** ("consider the expected value of information vs. its cost before probing")
- Optionally: self-reflection / allow revision

This tests whether prompting closes the optimality gap (E5).

---

## 7. Models

- Standard models (GPT-4o, Claude, Llama-70B, Qwen)
- Reasoning models (o1/o3, DeepSeek-R1)
- **Key question:** do reasoning models plan probes better, or just reason longer while probing suboptimally?

---

## 8. Experimental Procedure

1. Generate N problem instances per cell (e.g., N = 100–200 for statistical power).
2. For each: solve exact optimal policy offline → store optimal value, optimal probe count.
3. Run LLM with tool-calling loop; log full probe sequence + belief-relevant info + final choice.
4. Compute all metrics; compare distributions across conditions.

### Statistics
- Mixed-effects models: `regret ~ cost × prior_spread × model + (1|instance)`.
- Report effect sizes, CIs; correct for multiple comparisons.

---

## 9. Predicted / Interesting Findings

- LLMs achieve near-optimal **final choices** but **suboptimal probe counts** → decision quality masks meta-planning failure.
- **Cost insensitivity**: flat probe count across c (a clean, quotable result).
- **Uniform probing**: ignoring prior asymmetry, wasting probes on dominated arms.
- Reasoning models reduce regret mainly by better stopping, not better probe *selection*.

---

## 10. Tool Schema (concrete)

```
Tool: probe_arm(arm_id: int) -> reward_sample: float
  # Costs c per call

Tool: commit(arm_id: int) -> ends episode, returns true reward
```

System prompt states K, the prior, the cost c, and the objective explicitly (so the optimum is well-defined and the LLM has all info a rational agent needs).

---

# C3: Feature Acquisition Experiment — Full Design

## Overview

The LLM must classify a hidden object/instance by sequentially acquiring features (via tool calls), each with a cost, then commit to a class label. This is a **cost-sensitive sequential classification** task with a computable optimal policy.

---

## 1. Task Formalization

**Setup**
- Hidden true class $y \in \{1, ..., K\}$ drawn from prior $P(y)$
- Set of features $F = \{f_1, ..., f_M\}$, each observable via a tool call
- Each feature $f_i$ has:
  - Acquisition **cost** $c_i$
  - Conditional distribution $P(f_i \mid y)$ (defines its diagnosticity)
- LLM sequentially chooses: **acquire feature $f_i$**, or **stop and classify as $\hat{y}$**

**Payoff**
$$
U = R(\hat{y}, y) - \sum_{i \in \text{acquired}} c_i
$$
- $R(\hat{y}, y)$ = reward for correct (e.g., +100) / penalty for wrong (e.g., 0 or −50)

---

## 2. The Normative Optimum

This is a **metalevel POMDP** where belief state = posterior $P(y \mid \text{observed features})$.

- **State:** current posterior over classes
- **Actions:** acquire any un-acquired feature, or terminate with best class
- **Solve via:** backward induction / value iteration over belief states
- **Optimal terminal decision:** $\hat{y} = \arg\max_y P(y \mid \text{obs})$ (Bayes)
- **Optimal acquisition:** acquire feature maximizing **expected value of information** minus cost

> Keep $M \le 6–8$ features and $K \le 4–5$ classes so exact solution is tractable. Discretize/binary features to keep belief space manageable.

**Benchmarks to compute:**
| Policy | Description |
|--------|-------------|
| Optimal | Exact metalevel POMDP solution |
| Myopic/greedy | Acquire max single-step info-gain feature; myopic stopping |
| Random acquisition | Random feature order, fixed budget |
| No-acquisition | Classify from prior only |
| Full-information | Acquire everything (upper bound on accuracy, not utility) |

---

## 3. Independent Variables (Manipulations)

### IV1: Feature Cost
- Levels: low / medium / high (uniform), plus a **heterogeneous** condition (features differ in cost)
- **Tests:** cost sensitivity (E2) + whether LLM prefers cheap features rationally

### IV2: Feature Diagnosticity Structure
- **Uniform:** all features equally informative
- **Skewed:** one/two highly diagnostic features + several weak ones
- **Cost–value correlation:** informative features are expensive (rational tradeoff required)
- **Tests:** Does the LLM identify and prioritize high-value features?

### IV3: Prior Skew
- Uniform prior vs. skewed prior
- **Tests:** Does the LLM integrate priors (should acquire *less* when prior is confident)?

### IV4: Reward/Penalty Asymmetry
- Symmetric vs. high penalty for errors
- Optimal response: acquire *more* when errors are costly
- **Tests:** stakes sensitivity

### IV5 (optional): Semantic wrapping
- Abstract ("object X, feature A") vs. semantic ("animal, has_fur") with matched structure
- **Cross-context comparison** (ties into C1 vs C4 story)

---

## 4. Dependent Variables (Metrics)

**Optimality metrics**
- **Utility regret:** $U^* - U_{\text{LLM}}$ (primary)
- **Accuracy** of final classification
- **Number of acquisitions** vs. optimal

**Process metrics**
- **Acquisition quality:** avg. info-gain of chosen features vs. optimal ordering (rank correlation / precision@k)
- **Stopping error:** premature stops vs. over-acquisition (signed difference)
- **Cost-sensitivity slope:** regression of #acquisitions on cost level (compare slope to optimal)
- **Prior utilization:** change in acquisitions across prior conditions vs. optimal Δ

**Decision quality**
- **Bayes-consistency:** does $\hat{y}$ match posterior argmax given *actually observed* features? (isolates decision error from acquisition error)

**Metacognitive faithfulness** (ties to E4)
- Does stated CoT justification match the feature actually acquired?

---

## 5. Design Structure

- **Factorial:** IV1 (4) × IV2 (3) × IV3 (2) × IV4 (2) = 48 cells
  - (Trim if too large; a fractional/partial factorial focusing on IV1×IV2 as core is fine)
- **Instances per cell:** 30–50 randomized environments (different sampled true classes)
- **Models:** ≥4 (e.g., GPT-4-class, Claude, Llama-70B, a reasoning model like o1/o3)
- **Prompting conditions:** zero-shot vs. CoT (ties to E5)
- **Repetitions:** 3–5 per instance for stochasticity → report mean ± CI

---

## 6. Procedure (per trial)

1. Sample true class $y$; instantiate features from $P(f_i \mid y)$
2. Present LLM: task description, class set, available feature tools + costs, reward structure
3. LLM interacts turn-by-turn:
   - Tool call → return sampled feature value
   - Repeat until LLM emits terminal classification
4. Log: full acquisition sequence, CoT at each step, final decision, timing
5. Compute utility, compare to optimal policy on **same instance**

---

## 7. Analyses & Hypotheses

| Hypothesis | Test |
|-----------|------|
| H1: LLMs are sub-optimal but above myopic | Utility regret vs. baselines, paired tests |
| H2: LLMs are cost-insensitive | Compare acquisition-cost slope to optimal (interaction test) |
| H3: LLMs under-use priors | Prior condition × acquisitions ANOVA |
| H4: Decision (Bayes) error < acquisition error | Decompose regret into acquisition vs. decision components |
| H5: Reasoning models improve decisions more than acquisition | Model × metric interaction |

**Key decomposition (headline analysis):**
$$
\text{Total regret} = \underbrace{\text{acquisition regret}}_{\text{wrong features/stopping}} + \underbrace{\text{decision regret}}_{\text{non-Bayesian classification}}
$$
This separation is a clean, novel, and interpretable contribution.

---

## 8. Controls & Validity

- **Randomize** feature labels/order to prevent positional bias
- **Counterbalance** class-label naming
- **Sanity check:** verify LLM understands the interface (comprehension trials)
- **Leakage control:** in semantic version, ensure world-knowledge can't shortcut acquisition (or measure it explicitly)
- **Statistics:** mixed-effects models (random effects: instance, model) for regret; report effect sizes + CIs, correct for multiple comparisons

---

## Suggested Figures
1. Regret bar plot: LLMs vs. all baselines, per model
2. Acquisitions vs. cost line plot: LLM vs. optimal slope
3. Regret decomposition (stacked: acquisition vs. decision)
4. Feature-choice quality heatmap across diagnosticity conditions



---

# C4 Experiment Design: Diagnostic Decision-Making

## Overview

A medical-style diagnostic task where the LLM sequentially orders "tests" (tool calls) to identify a hidden disease, then commits to a diagnosis. This is the **bridge context**: naturalistic surface, but with an exactly computable optimal metalevel policy because *you* define the generative model.

---

## 1. Task Structure

### The Generative Model (POMDP)
- **Hidden state:** true disease $d \in \{d_1, ..., d_K\}$, sampled from prior $P(d)$
- **Tests (tools):** each test $t_j$ reveals a symptom/result $s_j$ with known likelihood $P(s_j \mid d)$
- **Test cost:** $c_j$ (fixed or variable per test)
- **Terminal action:** diagnose $\hat{d}$; reward $R(\hat{d}, d)$ (e.g., +1 correct, penalty matrix for errors)

### Agent's Decision Loop
At each step the LLM chooses:
1. **Order test $t_j$** → observe $s_j$, pay $c_j$
2. **Stop and diagnose** $\hat{d}$

**Objective:** maximize expected reward − total test cost.

---

## 2. Computing the Optimal Policy (Your Ground Truth)

Since you define $P(d)$, $P(s\mid d)$, and costs:

- **Belief state** = posterior over diseases (Bayesian update after each test)
- **Optimal metalevel policy** via backward induction on belief-state MDP
- For tractability: keep **K diseases small (3–6)** and **test count moderate (5–10)** so exact solution is feasible
- Precompute for each reachable belief: optimal action (which test, or stop + which diagnosis) and its Q-value

This gives per-step normative benchmarks, not just final-outcome comparison.

---

## 3. Experimental Conditions (Manipulations)

### Manipulation A: Test Cost
- Low / Medium / High uniform cost
- **Prediction (optimal):** fewer tests as cost rises → tests cost-sensitivity (E2)

### Manipulation B: Prior Skew
- Uniform prior vs. skewed prior (one disease likely)
- Optimal: skewed prior → fewer tests needed (strong prior = high initial certainty)
- Tests whether LLM integrates base rates (base-rate neglect is a classic human bias)

### Manipulation C: Test Informativeness
- Include highly diagnostic vs. weakly diagnostic tests
- Optimal: prioritize high-information tests early
- **Measures test-selection quality**, not just quantity

### Manipulation D: Error Asymmetry (optional, high-value)
- Asymmetric penalty matrix (missing disease A is very costly)
- Optimal: acquire more evidence before ruling out high-stakes disease
- Tests **utility-sensitive** information gathering

---

## 4. Semantic Variants (The Novel Comparison)

Run **three matched versions with identical underlying structure**:

| Variant | Surface | Purpose |
|---------|---------|---------|
| **V1 Abstract** | "Disease X," "Test 3" (no semantics) | Pure structural baseline |
| **V2 Plausible** | Real diseases + realistic test-symptom links | Semantics *aligned* with structure |
| **V3 Misleading** | Real names but test-likelihoods scrambled vs. real-world | Tests if priors override given evidence |

**Core research question:** Does medical semantics help LLMs plan better (V2 > V1), or do surface associations override the actual provided likelihoods (V3 < V1)?

*This isomorphic-structure design is the paper's rigor centerpiece.*

---

## 5. Baselines

| Baseline | Behavior |
|----------|----------|
| Optimal | Backward-induction policy (upper bound) |
| Random | Random test / random stop |
| Myopic (greedy VOI) | One-step lookahead info gain |
| Full-information | Always buy all tests then decide |
| Cost-blind | Maximize accuracy ignoring cost |

Myopic vs. optimal comparison is especially informative: LLMs often approximate greedy behavior.

---

## 6. Metrics

**Outcome-level**
- Expected reward vs. optimal (normalized regret)
- Diagnostic accuracy

**Process-level (the differentiator)**
- Number of tests vs. optimal
- Test-selection quality: fraction of chosen tests matching optimal action at that belief
- Stopping error: premature stop rate vs. over-acquisition rate
- Cost elasticity: Δ(tests) / Δ(cost) vs. optimal elasticity

**Metacognitive (links to E4)**
- Reasoning–action consistency: does stated rationale match chosen test?
- Calibration: stated confidence vs. actual posterior at stopping

---

## 7. Models & Protocol

- **Models:** ≥2 families × ≥2 sizes + reasoning models (o1/o3-style) for the emergence story
- **Prompting:** zero-shot vs. CoT vs. explicit-VOI prompt (crosses with E5)
- **Trials:** many sampled environments per condition (e.g., 100+) for statistics
- **Randomization:** shuffle test labels/order to prevent positional heuristics
- **Stats:** mixed-effects models (random effects for environment & model) on regret and test-count deviation

---

## 8. Hypotheses / Expected Findings

- **H1:** LLMs achieve near-optimal *accuracy* but sub-optimal *cost efficiency* (over-testing).
- **H2:** LLMs are cost-insensitive relative to optimal (flat elasticity).
- **H3:** V2 semantics improve selection; V3 misleading semantics degrade below abstract → surface-prior override.
- **H4:** Reasoning models improve test *selection* but not *stopping*.
- **H5:** Base-rate neglect: LLMs under-use skewed priors (over-test when prior already informative).

---

## 9. Concrete Micro-Example

```
Diseases: {Flu (0.5), Strep (0.3), Mono (0.2)}
Tests: {Throat swab (c=2), Blood test (c=5), Temperature (c=1)}
Likelihoods P(result | disease): [defined table]
Reward: +10 correct, -20 miss Mono, -5 miss others
```
Optimal policy (precomputed): "Take Temperature first; if high, take Throat swab; stop and diagnose Strep if positive..." — you compare the LLM's trajectory against this tree.

---

# C5: Investigative/Detective Scenario — Experimental Design

## Overview

The LLM plays an investigator identifying the correct culprit/cause. Clues are hidden behind tool calls, each with an acquisition cost. The LLM must decide **which clues to gather, when to stop, and whom to accuse**. This context naturally frames the *value of information* and *optimal stopping* problems in an engaging, interpretable way.

---

## 1. Environment Formalization

### State & Hypothesis Space
- **Hypotheses** $H = \{h_1, ..., h_K\}$: candidate culprits (e.g., 4 suspects)
- **True culprit** $h^*$: sampled from prior $P(H)$
- **Clues** $C = \{c_1, ..., c_N\}$: pieces of evidence behind tool calls

### Clue-Evidence Model (the key to a computable optimum)
Each clue $c_i$ has a **likelihood profile**: $P(c_i = e \mid h_k)$

Define each clue's outcome as a signal that updates beliefs via Bayes:
$$P(h_k \mid \text{observed clues}) \propto P(h_k)\prod_i P(c_i \mid h_k)$$

- **Diagnostic clues:** strongly discriminate between suspects
- **Weak clues:** minor belief shifts
- **Redundant clues:** correlated with already-gathered evidence (test whether LLM avoids them)

### Costs & Rewards
- Each tool call costs $\lambda$ (vary across conditions)
- Correct accusation: reward $R^+$
- Wrong accusation: penalty $R^-$
- **Net utility** = accusation payoff − total clue cost

---

## 2. Computing the Optimal Policy

Because clues are conditionally independent given $h^*$, this is a **sequential Bayesian experimental design / optimal stopping problem**:

- **Metalevel MDP state** = current belief distribution over $H$ + set of unqueried clues
- **Actions** = query clue $c_i$ (cost $\lambda$) OR stop-and-accuse $h_k$
- Solve via **backward induction** over the belief-state tree (feasible for small $N$, $K$)
- Optimal policy yields: which clue to query given current belief, and the optimal stopping/accusation rule

> **Keep N ≤ 6–8 clues, K ≤ 4 suspects** for exact computation. For larger, use myopic VOI (single-step lookahead) as an approximate benchmark and note the gap.

---

## 3. Tool Interface

```
Tools available:
- examine_clue(clue_name) → returns evidence text/value, costs λ
- list_available_clues() → clue names + descriptions (free)
- make_accusation(suspect) → ends episode
```

The LLM sees: the scenario setup, suspect list, cost per clue, and reward structure.

---

## 4. Experimental Conditions (Manipulations)

| Factor | Levels | Tests |
|--------|--------|-------|
| **Clue cost** $\lambda$ | low / medium / high | Cost sensitivity (E2) |
| **Prior structure** | uniform / skewed | Do LLMs exploit priors? |
| **Diagnosticity layout** | diagnostic-early / diagnostic-late | Strategic ordering (à la Mouselab variance manip.) |
| **Redundancy** | present / absent | Do LLMs avoid uninformative queries? |
| **Semantic realism** | narrative-rich / neutral-labeled | Do surface semantics help/hurt? |

Cross key factors → factorial design. Generate many randomized instances per cell.

---

## 5. Metrics

### Primary
1. **Value gap:** achieved net utility vs. optimal-policy utility
2. **Accusation accuracy** vs. optimal accuracy (decision quality)
3. **Information efficiency:** clues gathered vs. optimal number

### Process (the novel contribution)
4. **Query-choice alignment:** at each step, did the LLM pick the max-VOI clue? (per-step optimality rate)
5. **Stopping calibration:** stopped too early (under-investigation) vs. too late (over-investigation)
6. **Redundancy rate:** fraction of low-VOI/redundant clues queried
7. **Belief calibration:** if you elicit confidence, compare to true posterior after observed clues

### Metacognitive (optional, high-value)
8. **Reasoning–action consistency:** does stated rationale for a clue match its actual VOI ranking?

---

## 6. Baselines

- **Optimal metalevel policy** (upper bound)
- **Myopic/greedy VOI** (single-step lookahead — human-like heuristic)
- **Random query** policy
- **Query-everything** then decide (cost-insensitive)
- **Immediate accusation** from prior (zero information)

Positioning LLMs among these baselines gives an interpretable "**where on the rationality spectrum**" story.

---

## 7. Models

- Range across families/sizes + reasoning models (o1/o3-style)
- Prompting conditions: zero-shot, CoT, explicit-VOI-prompt

---

## 8. Predicted / Interesting Findings

- LLMs likely **over-investigate at high cost** (cost-insensitivity)
- LLMs may **query redundant clues** (fail conditional-independence reasoning)
- **Narrative-rich framing** may improve engagement but introduce **suspect biases** (e.g., accusing the "suspicious-sounding" character) → compelling qualitative failure cases
- Reasoning models improve *accuracy* more than *information efficiency*

---

## 9. Example Instance (Concrete)

> **Scenario:** Museum theft. Suspects: {Curator, Guard, Visitor, Intern}. True culprit: Guard.
>
> **Clues (with likelihoods encoded):**
> - `alibi_check` (diagnostic: rules out 2 suspects)
> - `fingerprints` (diagnostic)
> - `motive_interview` (weak)
> - `security_footage` (redundant with fingerprints)
> - `financial_records` (moderate)
>
> Cost per clue = 5; correct = +100, wrong = −50.
>
> **Optimal policy:** query `alibi_check` → `fingerprints`, then accuse. Expected 2 clues.
> **Suboptimal LLM behavior to detect:** querying all 5, or accusing after `motive_interview` alone.

---

## Design Strengths for Publication
- **Exact optimum** (small, conditionally-independent structure)
- **Engaging, interpretable** failure cases (great for qualitative analysis)
- **Matched to abstract contexts** (can build isomorphic version of C1)
- Tests **VOI reasoning, stopping, prior use, redundancy avoidance** all at once

---

# C6 Experiment Design: Research / Literature Search

This context tests whether LLMs optimally acquire information in a realistic RAG-style agentic workflow, while retaining a **constructed ground-truth optimum**.

---

## 1. Task Overview

The LLM must answer a research question by selectively acquiring information sources at increasing cost, then commit to an answer.

**Tool hierarchy (increasing cost, increasing information):**

| Tool | Cost | Information Returned |
|------|------|---------------------|
| `search(query)` | Low | List of paper titles + 1-line snippets |
| `read_abstract(id)` | Medium | Full abstract |
| `read_full(id)` | High | Key findings / detailed content |

The LLM chooses **which** sources, **how deep** to read, **when to stop**, and produces a **final answer**.

---

## 2. Constructing Ground Truth (The Critical Part)

To compute the optimal metalevel policy, you **generate the environment**, not scrape real papers.

### Generative structure
1. Define a **latent answer** `A*` (e.g., which of K methods performs best).
2. Create N synthetic "papers," each a **noisy evidence source** about `A*`.
   - Each paper `i` has an informativeness parameter and a stance/finding.
   - Deeper reads reveal evidence with **lower noise** (abstract = noisy signal, full text = precise signal).
3. This defines a **Bayesian belief-updating process**: each tool call updates the posterior over `A*`.

### Optimal policy
- State = current posterior over `A*` + which sources read at which depth.
- Actions = acquire next source/depth, or stop and answer.
- Reward = (correctness payoff) − (accumulated acquisition cost).
- **Solve via backward induction** (small N) or **BMPS/myopic VOI approximation** (larger N), reported with the approximation clearly stated.

> This makes C6 *look* naturalistic but keeps a computable normative benchmark—the core methodological claim.

---

## 3. Experimental Conditions

### Primary manipulations
- **Cost regime:** low / medium / high tool cost (→ tests cost sensitivity, E2)
- **Evidence dispersion:** concentrated (one paper is decisive) vs. distributed (need many weak signals)
  - Optimal strategy differs sharply: targeted deep-read vs. broad shallow-scan
- **Prior informativeness:** vague vs. strong prior on `A*`
  - Tests whether LLM rationally stops early when prior is strong

### Semantic condition (key comparison)
- **Realistic wrapper:** plausible ML/science topic where the LLM's real knowledge *could* bias it
- **Neutral wrapper:** structurally identical but semantically empty
- → Tests whether world-knowledge priors help or produce **premature stopping** (answering from prior without acquiring evidence)

---

## 4. Metrics

**Optimality**
- Achieved reward vs. optimal-policy reward (regret)
- Vs. baselines: random, greedy-VOI, read-everything, answer-immediately

**Process quality**
- # tool calls vs. optimal # tool calls
- Depth profile: does it read full-text when abstracts suffice? (over-acquisition)
- Source selection: does it query high-VOI sources first?

**Stopping**
- Premature-stop rate vs. over-search rate
- Correlation between posterior confidence and stopping (rational agents stop when confident)

**Faithfulness (links to E4)**
- Does stated reasoning ("I need more evidence on X") match the next tool call?

---

## 5. Design Table

| Factor | Levels |
|--------|--------|
| Cost regime | Low / Med / High |
| Evidence dispersion | Concentrated / Distributed |
| Prior strength | Weak / Strong |
| Semantic wrapper | Realistic / Neutral |
| Model | (GPT / Claude / Llama / reasoning models) |

Full factorial on environment factors × models, with **many randomized instances per cell** for statistical power. Report mixed-effects models with instance as random effect.

---

## 6. Predicted / Hypothesized Findings

These make the paper compelling if confirmed:

1. **Cost insensitivity:** LLMs over-read at high cost (don't cut back like optimal agent).
2. **Depth bias:** systematic preference for `read_full` when `read_abstract` is optimal (or the reverse—"lazy" shallow reading).
3. **Semantic override:** in the realistic wrapper, LLMs stop early and answer from prior knowledge—hurting optimality relative to neutral wrapper.
4. **Reasoning models** search *longer* but not *smarter* (more calls, not higher VOI-per-call).

---

## 7. Confounds to Control

- **Answer leakage:** ensure the correct answer is *not* derivable from the question or search titles alone (force genuine acquisition).
- **Context-length effects:** keep returned content length matched across depths to isolate information value from token burden.
- **Tool-format familiarity:** use standard function-calling schemas; include a warm-up example so failures reflect planning, not formatting.
- **Position effects:** randomize source ordering.

---

# C7: Purchasing/Comparison Decision Experiment

A complete, publication-ready design with computable ground-truth optima.

---

## 1. Task Overview

The LLM acts as a **shopping agent** choosing one product from a set. Product attributes are hidden behind tool calls; each query costs. The agent decides **which attributes to reveal, when to stop, and which product to buy** to maximize expected utility net of query costs.

**Why this context works:**
- Utility function is explicit → value-of-information is exactly computable
- Naturalistic and intuitive (strong reviewer appeal)
- Can be made isomorphic to abstract reward trees (C1) for the semantics comparison

---

## 2. Environment Formalization

### Setup
- **N products** (e.g., N = 4)
- **K attributes** per product (e.g., price, quality, durability, reviews, warranty)
- Each attribute value drawn from a known distribution (revealed to the agent as priors)
- **Utility function** (given to the agent):

$$U(\text{product}_i) = \sum_{k=1}^{K} w_k \cdot v_{i,k}$$

where $w_k$ are known attribute weights, $v_{i,k}$ are attribute values.

### Tools
- `reveal(product_i, attribute_k)` → returns $v_{i,k}$, incurs cost $c$
- `purchase(product_i)` → terminates episode, yields $U(\text{product}_i)$

### Agent's objective
$$\max \; \mathbb{E}[U(\text{chosen product})] - c \cdot (\text{number of reveals})$$

---

## 3. Computing the Normative Optimum

This is a **metalevel MDP**:
- **Belief state:** which cells revealed + their values + posterior over unrevealed cells
- **Meta-actions:** reveal any hidden cell, or stop and purchase best-expected product
- **Reward:** terminal utility minus accumulated query cost

**Solution methods (by scale):**
| Environment size | Method |
|------------------|--------|
| Small (e.g., 3×3, few discrete values) | Exact backward induction over belief states |
| Medium | BMPS (Bayesian myopic policy) / value-of-computation approximation |
| Large | Monte Carlo rollout policy as near-optimal reference |

**Baselines to report:**
- Optimal metalevel policy (ceiling)
- Full-information (reveal everything, then choose) — measures over-collection cost
- Zero-information (choose by priors only) — measures under-collection
- Myopic VOI (greedy one-step value of information)
- Random reveal + threshold stop

---

## 4. Experimental Manipulations

### M1. Query Cost (→ tests E2, cost sensitivity)
- Levels: $c \in \{0, \text{low}, \text{medium}, \text{high}\}$
- **Prediction:** optimal # reveals decreases monotonically. Test if LLM tracks this.

### M2. Attribute Weight Structure (→ tests E3, structure exploitation)
- **Concentrated:** one attribute dominates (rational agent should query it first)
- **Uniform:** all attributes matter equally
- **Deceptive:** the salient-sounding attribute (e.g., "price") has low weight
- Tests whether LLM uses **weights** rationally or follows **semantic salience**

### M3. Prior Informativeness (→ VOI structure)
- **High-variance priors:** information is valuable (should query more)
- **Low-variance priors:** little to learn (should query less / stop early)

### M4. Semantic Framing (→ the key novel contribution)
- **Semantic:** real product names, realistic attributes ("noise-canceling headphones: battery life, sound quality...")
- **Abstract-isomorphic:** same numbers, neutral labels ("Option A, Attribute 1")
- **Misleading-semantic:** attribute names imply importance that contradicts $w_k$
- **Tests:** Do semantics help (better priors), hurt (heuristic override), or wash out?

---

## 5. Metrics

### Primary
- **Utility regret:** $U^*_{\text{optimal policy}} - U_{\text{LLM}}$
- **Query efficiency:** # reveals relative to optimal
- **Decision accuracy:** P(chose highest-utility product)

### Process-level (the Tool-Lab distinctive value)
- **Acquisition targeting:** fraction of reveals on high-weight attributes vs. optimal
- **Stopping calibration:** stop-time vs. optimal stop-time (over/under-collection)
- **Cost elasticity:** slope of (# reveals) vs. cost, compared to optimal slope
- **First-query rationality:** is the first reveal the highest-VOI cell?

### Metacognitive (→ E4)
- **Reasoning–action consistency:** does CoT-stated plan match executed reveals?

---

## 6. Design Matrix

Recommended factorial:

```
4 cost levels × 3 weight structures × 2 prior levels × 4 framings
= 96 cells
× 30 environment instances per cell
× M models
```

Randomize product/attribute assignment; counterbalance to avoid position bias.

---

## 7. Statistical Analysis

- **Mixed-effects models:** regret ~ cost × structure × framing + (1|instance) + (1|model)
- **Cost-sensitivity test:** interaction of (LLM vs. optimal) × cost on # reveals
- **Bias decomposition:** partition regret into over-collection vs. under-collection vs. wrong-target vs. wrong-decision
- Report human baselines if feasible (from Mouselab literature or a small human study) — strengthens cog-sci framing

---

## 8. Hypotheses / Predicted Story

| Hypothesis | Prediction |
|-----------|-----------|
| H1 | LLMs achieve high *decision accuracy* but sub-optimal *query efficiency* |
| H2 | LLMs are **cost-insensitive** (over-query when queries are expensive) |
| H3 | LLMs follow **semantic salience** over true weights (fail in deceptive condition) |
| H4 | Semantic framing helps priors but hurts when misleading |
| H5 | Reasoning models improve targeting but not stopping calibration |

---

## 9. Controls & Confounds to Preempt

- **Position/order bias:** randomize product ordering
- **Numeric parsing:** verify LLM can compute $U$ correctly when given full info (sanity check)
- **Prompt leakage:** ensure optimal answer isn't inferable from phrasing
- **Cost comprehension check:** confirm model acknowledges the cost in reasoning

---

# Question

Review the proposed paper for ACL.