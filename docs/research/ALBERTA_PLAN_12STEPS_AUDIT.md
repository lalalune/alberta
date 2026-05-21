# The Alberta Plan for AI Research: Complete 12-Step Analysis
**Paper:** Sutton, Bowling, Pilarski (2023) — arXiv:2208.11173
**Date Extracted:** 2026-05-21

---

## Executive Summary: Architectural Vision

The Alberta Plan is a **fundamental, long-term research roadmap** (5–10 year horizon) toward **continual, model-based AI**. It differs from standard RL in three core ways:

1. **Temporal Uniformity**: No special training phases, offline data, or episodic boundaries. All learning, planning, and meta-learning happen on *every* time step in a continuing, non-stationary setting.
2. **Computational Realism**: Methods must scale with compute (deep learning, search) and respond in real-time. Reaction time matters as much as optimality.
3. **Experiential Grounding**: Only raw observations, actions, and rewards—no human labels, domain knowledge, or access to environment internals.

The complete system is a **hierarchical agent** with:
- **Foreground** (real-time, every step): Perception → State → Reactive Policy → Action → Value functions (learning)
- **Background** (asynchronous, planning): World model, search, temporal abstraction

---

## The 12 Steps: Detailed Breakdown

### Step 1: Representation I — Continual Supervised Learning with Given Features

**Exact Name & Description:**
> "Continual supervised learning with given features. Step 1 is exemplary of the primary strategy of the Alberta Plan: to focus on a particular issue by considering the simplest setting in which it arises and attempting to deal with it there, fully, before generalizing to more complex settings."

**Core Problem Statement:**
- Infinite stream of **(x_t, y_t*)** pairs; learner produces **y_t = w_t^T x_t + b_t**
- Goal: minimize squared error **E[(y_t* − y_t)²]**
- Non-stationarity: **y_t* = (w_t*)^T x_t + b_t* + η_t** where **w_t*, b_t* drift over time** and **distribution of x_t shifts**

**Key Algorithmic Requirements:**
1. **Per-feature adaptive step-sizes α_i^t** (not global α)
   - Different learning rates for relevant vs. irrelevant features
   - Enables feature relevance discovery without changing the feature set
2. **Meta-learning algorithms for step-size adaptation**
   - Replace hand-tuned hyperparameters with principled algorithms
   - Must work across diverse problems with no domain knowledge
3. **Online normalization** of inputs
   - Non-stationary tracking of mean μ_i^t and std σ_i^t
   - Transform inputs: **x̃_i^t = (x_i^t − μ_i^t) / σ_i^t**
   - Effect on learning efficiency is still unexplored in literature
4. **Temporal uniformity**: Weight and trace updates on every step via SGD

**Comparison Set (Footnote 11 Methods):**
- **Adaptive**: IDBD (Sutton 1992), Autostep (Mahmood et al. 2012), Autostep-for-GTD(λ) (Kearney et al. 2022), Auto (Degris in prep.—unpublished)
- **General-purpose**: Adam (Kingma & Ba 2014), RMSprop (Tieleman & Hinton 2012)
- **Normalization**: Batch Normalization (Ioffe & Szegedy 2015)
- **Classical**: NADALINE (Sutton 1988)

**Success Criteria (What "Done" Looks Like):**
> "The overall idea of Step 1 is to design as powerful an algorithm as possible given a fixed feature representation. It should include all the most important issues of non-stationarity in the problem (for a fixed set of linear features), including the tracking of changes in feature relevance."

Specific evidence targets:
- Tuning-free step-size adaptation beats manually tuned LMS across diverse non-stationary problems
- Per-feature meta-learning successfully isolates relevant vs. irrelevant features
- Online normalization measurably improves robustness to input distribution shifts
- Algorithm works on sparse-relevance tasks, drifting weights, drifting bias, and input-distribution shift

**Robot/Real-World Requirements:**
- Single-step API: **predict(observation) → value** and **update(error, features)** per time step
- No episodic resets, no replay buffer, no offline training phase
- Sub-millisecond latency per update (daemon-compatible)

**Referenced Papers & Basis:**
- IDBD: Sutton (1992) "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"
- Autostep: Mahmood et al. (2012) "Tuning-free Step-size Adaptation: Problem-Invariant Performance"
- Adam: Kingma & Ba (2014)
- RMSprop: Tieleman & Hinton (2012)
- Batch Normalization: Ioffe & Szegedy (2015)

---

### Step 2: Representation II — Supervised Feature Finding

**Exact Name & Description:**
> "Supervised feature finding. This step is focused on creating and introducing new features (made by combining existing features) in the context of continual supervised learning as in Step 1, except now targets will be vectors y_t* approximated by output vectors y_t."

**Core Problem Statement:**
- Multi-task: each component of **y_t** targets a separate task **y_t*_j**
- Features are now **learned** (not fixed)
- Feature construction: **nonlinear combinations** of existing features
- Bounded budget: can only maintain **N_active** features in parallel
- Non-stationary: feature utility and relevance change over time

**Key Algorithmic Requirements:**
1. **Smart feature generation**: Create promising new features from existing ones
   - Product features (interactions)
   - Compositional/recursive features (features of features)
   - Feature-generation algorithm should use prior experience and current utility signals
2. **Candidate testing & ranking**: Evaluate new features before full promotion
   - Online, single-pass validation (no holdout splits)
   - Utility assignment that accounts for future feature effects
3. **Resource management**: Maintain bounded feature budget
   - Identify weak features for deletion
   - Make room for new candidates
   - Learnable replacement/deletion policies
4. **Utility tracking across tasks**
   - Global utility (impact on all outputs)
   - Future utility estimation (how useful will this feature be next?)
5. **Fair nonlinear baselines**: MLPs with equivalent capacity

**Success Criteria:**
> "The point of this step is to explore the challenging issues in managing a limited resource for representing and learning about features. You can represent and gather data on a limited number of features. When should you discard an old feature so that you can collect data on a new one? How is the new feature constructed? How is the discarded feature selected?"

Specific evidence targets:
- Feature-construction methods outperform fair MLPs on out-of-hypothesis-class tasks (polynomial, frequency mismatch, compositional)
- Utility tracking successfully ranks/discards weak features without catastrophic forgetting
- Online feature budget management avoids resource exhaustion
- Feature discovery works without offline batch training or replay

**Robot/Real-World Requirements:**
- Continuous multi-task environment (vector rewards/targets)
- State/task switching without episodic resets
- Feature utility must be computable from online streaming data only

**Referenced Papers & Basis:**
- GVF foundations: Sutton et al. (2011) "Horde: A Scalable Real-Time Architecture"
- Multi-task learning: Implied from Step 1 extensions
- Feature construction: Product features, compositional DAGs (research direction, not cited prior art)

---

### Step 3: Prediction I — Continual GVF Prediction Learning

**Exact Name & Description:**
> "Prediction I: Continual GVF prediction learning. Repeat the above two steps for sequential, real-time settings where the data is not i.i.d., but rather is from a process with state and the task is generalized value function (GVF) prediction. First with given linear features, then with feature finding."

**Core Problem Statement:**
- Data is now **sequential with state** (Markov process), not i.i.d.
- GVF prediction: estimate **V_π,γ,r,z(s)** — the cumulative discounted non-reward signal under policy π with discount γ and cumulant r
  - **V_π,γ,r,z(s) = E[Σ γ^t r_t + γ^τ z(s_τ)]** where τ is pseudo-termination time
- Multi-step returns via eligibility traces
- Off-policy learning (if possible)
- Recurrent/reactive networks with **limited computation per observation**

**Key Algorithmic Requirements:**
1. **GVF architecture**: Four question functions per demon
   - **Policy π**: which actions to follow (fixed for prediction, greedy for control)
   - **Pseudo-discount γ**: when to reset traces
   - **Cumulant r**: what signal to accumulate (not just reward)
   - **Terminal value z**: value at pseudo-termination
2. **Eligibility traces** with per-demon trace decay
3. **Temporal credit assignment** via TD learning (not one-step supervised)
4. **Feature discovery under TD targets** (new challenge vs. Step 2)
   - Features that minimize TD error, not supervised loss
   - Trace-based utility signals
5. **Perception/state construction**
   - Recurrent networks allowed
   - Minimal processing per step

**Success Criteria:**
> "Ideally, this would be taken all the way to off-policy learning. Ideally this would be in a real-time setting with recurrent networks that do a limited amount of processing per observation. Here we explicitly address the question of constructing state, the perception part of the standard agent model."

Specific evidence targets:
- GVF prediction beats supervised baselines on tasks with temporal structure
- Per-demon γ/λ enables independent trace decay strategies
- Feature discovery works under TD targets (harder than supervised)
- Off-policy GVF learning converges with function approximation

**Robot/Real-World Requirements:**
- Stateful interaction: current observation depends on history
- Predictive state construction: recurrent processing
- Real-time constraint: O(1) or O(log n) computation per step

**Referenced Papers & Basis:**
- GVF core: Sutton et al. (2011) "Horde" & Sutton (1988) on temporal differences
- Conditional testbeds: Classical benchmarks (conditional expectation prediction)
- Off-policy: Sutton & Barto (2018) GTD/TDC algorithms
- Traces: Sutton & Barto (2018) Chapter 13 on eligibility traces

---

### Step 4: Control I — Continual Actor-Critic Control

**Exact Name & Description:**
> "Control I: Continual actor-critic control. Repeat the above three steps for control. First in a conventional k-arm bandit setting, then in a contextual bandit setting with discrete softmax actions, then in a sequential setting with given features, and finally in a sequential setting with feature finding."

**Core Problem Statement:**
- Goal: maximize cumulative reward in bandit → contextual bandit → episodic → continuing settings
- Requires both:
  - **Critic**: GVF value function from Steps 1-3
  - **Actor**: policy learning to maximize the critic's value estimate

**Key Algorithmic Requirements:**
1. **Bandit control** (no state transitions)
   - k-arm setting: learn mean reward per arm
   - Contextual: condition on observation before arm selection
2. **Sequential control** with function approximation
   - Actor: policy update toward greedy(Q̂(s))
   - Critic: TD learning of Q-function
3. **Interaction between actor and critic**
   - Critic provides gradient signal for actor
   - Actor's policy affects critic's training data (correlated updates)
   - Both must learn continually and robustly
4. **Feature finding for control**
   - Features must be useful for both policy representation and value estimation
   - Feature ranking based on action/reward effects (more complex than Step 2)

**Sub-Steps (Progressive Complexity):**
1. k-arm bandit (no state)
2. Contextual bandit (observation conditioning, no transitions)
3. Sequential + given features
4. Sequential + feature finding

**Success Criteria:**
> "[Actor-critic] learning is continual and robust."

Specific evidence targets:
- Actor converges to near-greedy policies under non-stationary reward distributions
- Critic's TD error provides stable gradient for policy learning
- Feature sharing between actor and critic doesn't cause divergence
- Continual learning maintains off-policy stability across feature distributions

**Robot/Real-World Requirements:**
- Real-time control loop: perceive → compute policy → act → learn
- No episodic resets (continuing environments)
- Graceful handling of feature drift (utilities change)

**Referenced Papers & Basis:**
- Actor-critic: Sutton & Barto (2018)
- GVF critic: Steps 1-3 foundations
- Off-policy actor-critic: Implied from GVF off-policy work

---

### Step 5: Prediction II — Average-Reward GVF Learning

**Exact Name & Description:**
> "Average-reward GVF learning. The general idea here is to extend our general prediction learning algorithms for GVFs to the average-reward case. We separate the cumulant from the terminal value, and the cumulant is always the reward."

**Core Problem Statement:**
- Shift from **discounted** (γ < 1) to **average-reward** (undiscounted, continuing) objectives
- **Average reward ρ = lim_{T→∞} (1/T) Σ_t r_t**
- **Differential value V^ρ(s) = E[Σ_t (r_t − ρ) | s_t = s]** (reward surplus)

**Key Algorithmic Requirements:**
1. **Separate cumulant from terminal value**: r is always the reward, z becomes option-specific terminal value
2. **Two cases** (paper lists as relevant options):
   - **Case 1**: Learn differential value, with separate average-reward estimation
     - Average reward ρ estimated online
     - Feature stream normalized by subtracting ρ
     - No termination (continuing task)
   - **Case 2**: Learn conventional value + expected option duration
     - Used when combining with option termination
     - May be combined with Case 1
3. **Per-demon average reward**: Each GVF demon may have its own ρ
4. **Asynchronous DP**: Planning updates in any order (not full backup per step)

**Success Criteria:**
> "What we learned in the first four steps should carry over to the learning algorithms for average-reward GVFs for prediction and control with minimal changes."

Specific evidence targets:
- Algorithm converges to correct average reward ρ under non-stationarity
- Differential value learning is more stable than discounted learning in continuing tasks
- Feature ranking works under average-reward signal
- Off-policy extensions exist (for future Steps 6+)

**Robot/Real-World Requirements:**
- **Continuing tasks**: No episode boundaries, no terminal states
- **Real-time**: Compute ρ, differential TD error, and feature updates per step
- Long-run stationarity: Average reward well-defined over agent lifetime

**Referenced Papers & Basis:**
- Average-reward RL: Puterman & Brumelle; Bertsekas & Tsitsiklis (classical; not cited)
- GVF extension: Steps 1-4 foundations + average-reward generalization

---

### Step 6: Control II — Continuing Control Problems

**Exact Name & Description:**
> "Control II: Continuing control problems. We will need some continuing problems to test average-reward algorithms for learning and planning. Currently we have River Swim, Access-control Queuing, foraging problems like the Jellybean World, and GARNET."

**Core Problem Statement:**
- Create / adapt benchmark environments for **continuing (undiscounted)** control
- Most RL benchmarks are episodic; need non-terminal, long-running tasks

**Continuing Benchmarks Mentioned:**
1. **River Swim**: Navigation with rewarding states; tunable horizon
2. **Access-Control Queuing**: Job scheduling with service times
3. **Jellybean World**: Foraging / spatial navigation
4. **GARNET**: Stochastic benchmark generator (Bhatnagar et al.)

**Key Algorithmic Requirements:**
- Benchmark design must isolate continuing control issues:
  - Average reward well-defined and stable
  - Policies are asymptotically optimal (not episodic terminal value)
  - Non-stationarity (reward/dynamics shifts) testable

**Success Criteria:**
> "We will need some continuing problems to test average-reward algorithms."

Specific evidence targets:
- Algorithms trained on continuing benchmarks converge to stationary policies
- Average reward ρ is identifiable and learnable
- Off-policy and on-policy algorithms both work on same domains
- Feature ranking works in continuing setting

**Robot/Real-World Requirements:**
- Long-running agents: lifelong operation without resets
- Stationary asymptotic behavior: eventual policy optimality
- Non-stationary shifts: test robustness to environment changes

**Referenced Papers & Basis:**
- River Swim, GARNET: Bhatnagar et al. (not explicitly cited; benchmark references)
- Jellybean World: Foraging literature (Schmajuk, Ackley)

---

### Step 7: Planning I — Planning with Average Reward

**Exact Name & Description:**
> "Planning I: Planning with average reward. Develop incremental planning methods based on asynchronous dynamic programming for the average-reward criteria. The initial work here would be for the tabular case, but the case with function approximation should be close behind."

**Core Problem Statement:**
- Use learned world model to improve policies offline (asynchronously)
- No environment interaction during planning
- **Asynchronous DP**: Update states in any order, not sequential DP backup

**Key Algorithmic Requirements:**
1. **Average-reward value iteration**
   - **V(s) ← max_a [r(s,a) + Σ_{s'} P(s'|s,a) V(s')]** (adjusted for ρ)
   - Converges to differential value function
2. **Asynchronous updates**: States updated in arbitrary order (not full sweep)
   - Improves convergence speed in large state spaces
   - Requires careful sweep ordering (search control)
3. **Function approximation**: Extend tabular to linear/nonlinear
   - Must incorporate Steps 1-3 lessons (feature finding, normalization, step-size adaptation)
4. **Integration with value function learning** (Steps 1-5)
   - Critic's feature representation used for planning
   - Bi-directional: planning improves features, features improve planning

**Success Criteria:**
> "Develop incremental planning methods based on asynchronous dynamic programming for the average-reward criteria."

Specific evidence targets:
- Planning converges to ε-optimal average-reward policies
- Asynchronous order choice affects convergence speed (empirical analysis)
- Function approximation doesn't diverge under realistic non-stationarity
- Planning and learning operate in parallel (background process)

**Robot/Real-World Requirements:**
- Background planning: No real-time constraint (asynchronous)
- Model learning: World model learned from interaction (Step 8)
- No episodic boundaries: Planning must handle continuing tasks

**Referenced Papers & Basis:**
- Asynchronous DP: Bertsekas (1982) "Distributed dynamic programming"; Sutton & Barto (2018) Chapter 8
- Average-reward DP: Puterman & Brumelle (classical)
- Prioritized sweeping: Peng & Williams (1993), Moore & Atkeson (1993)
- Function approximation: Sutton et al. (2008) "Dyna-style Planning with Linear Function Approximation and Prioritized Sweeping"

---

### Step 8: Prototype-AI I — One-Step Model-Based RL

**Exact Name & Description:**
> "Prototype-AI I: One-step model-based RL with continual function approximation. Our first prototype-AI would be based on average-reward RL, models, planning, and continual non-linear function approximation. This would move beyond past work on Dyna by incorporating general continual function approximation, but would still be limited to one-step models."

**Core Problem Statement:**
- **First integrated prototype agent** (not just components)
- Combines Steps 1-7 into functioning system
- One-step world model: **E[r_{t+1}, s_{t+1} | s_t, a_t]**
  - No temporal abstraction (yet)
  - But model enables planning
- Average-reward criterion throughout

**Key Algorithmic Requirements:**
1. **Recursive state update** (Perception component)
   - Maintain state belief from observations and actions
   - Feature construction (Step 2) ranks and selects useful features
2. **One-step environment model**
   - Expectation model: **E[r, s' | s, a]** (deterministic or mean)
   - Sample model: draw **r, s'** from learned distribution
   - Or hybrid
3. **Feature finding with planning feedback**
   - Feature importance ranked by:
     - Learning/critic utility (Step 2)
     - Model prediction error (feature helps predict transitions)
     - Planning usefulness (feature helps value improvement)
   - Cycle: learning → feature ranking → planning → feature ranking
4. **Search control** (rough version)
   - Decide which states/actions to prioritize in planning
   - Simple variants: breadth-first, priority queue
   - May use "something like MCTS" (authors suggest, not require)
5. **Average-reward learning + planning loop**
   - Foreground: learn from interaction
   - Background: improve model, plan value updates

**Success Criteria:**
> "Without temporal abstraction Prototype-AI 1 will be weak and limited in many ways (and perhaps not all that impressive), but it will undoubtedly involve its own challenges. Or maybe it will be easy and non-impressive, in which case we can complete it and move on to Prototype-AI II."

The paper is honest: Prototype-AI I may not be "impressive" without options. Success means:
- Integrated system runs without divergence
- Planning + learning improve over model-free baseline
- Feature ranking accounts for model feedback
- One-step model accuracy is learnable

**Components Specified:**
- **(a)** Recursive state-update process ✓
- **(b)** One-step environment model ✓
- **(c)** Feature finding with model feedback ✓
- **(d)** Feature ranking for model and learning ✓
- **(e)** Cycle: model learning → feature ranking → planning → feature ranking ✓
- **(f)** Search control (sketchy) ✓

**Challenges Flagged:**
> "Sub-steps b, e, and f will involve challenging new issues not faced previously and may not be addressable in a fully satisfactory manner prior to temporal abstraction."

**Robot/Real-World Requirements:**
- Real-time agent loop: perceive → select action → learn → plan (background)
- World model sufficient for one-step prediction (limited lookahead)
- Continuing environment with average-reward feedback

**Referenced Papers & Basis:**
- Dyna: Sutton (1990) "Integrated Architectures for Learning, Planning, and Reacting"
- Model-based RL: Sutton & Barto (2018) Chapter 8
- Feature finding: Step 2 + model-importance extension (novel)
- MCTS: Suggestive reference only (not required)

---

### Step 9: Planning II — Search Control and Exploration

**Exact Name & Description:**
> "Planning II: Search control and exploration. In this second planning step we develop the control of planning. Planning is viewed as asynchronous value iteration with function approximation. Asynchronous value iteration allows the states to be updated in any order, but the order chosen greatly affects planning efficiency."

**Core Problem Statement:**
- **Search control** = order of state updates in asynchronous DP
- With function approximation, state ordering is even more critical (generalization effects)
- Generalize from tabular prioritization ideas (prioritized sweeping, small backups) to function-approximated settings

**Key Algorithmic Requirements:**
1. **Prioritized sweeping** generalized to function approximation
   - Select states with high TD error or utility first
   - Propagate improvements through learned value function
2. **Small backups** (Van Seijen & Sutton 2013)
   - One-step Bellman updates instead of full sweeps
   - Better integration with asynchronous planning
3. **Uncertainty-aware prioritization**
   - Give priority to states with uncertain value estimates
   - Consider aleatoric (stochastic) vs. epistemic (function approximation) uncertainty
4. **Exploration strategy**
   - How to balance exploitation of current best estimate vs. exploration of uncertain states?
   - Novel strategies: curiosity-driven, information-gain, optimism-under-uncertainty

**Search Control Spectrum (Authors' Framing):**
- Tabular asynchronous DP (deterministic order)
- Prioritized sweeping (highest-priority-first)
- Classical heuristic search (A*, Dijkstra)
- Monte Carlo Tree Search (MCTS)
- Random exploration

**Success Criteria:**
> "With function approximation the effect is even greater. Early efforts to control the planning process include prioritized sweeping and small backups... [generalize to] take into account the uncertainty of various parts of the model."

Specific evidence targets:
- Planned updates with good prioritization improve policy faster than tabular DP
- Model uncertainty correlates with value update significance
- MCTS or greedy prioritization outperforms random planning
- Exploration efficiently discovers high-value states

**Robot/Real-World Requirements:**
- Anytime planning: Can be interrupted; better solutions as time allows
- Computational budget: Decide allocation of planning steps
- Model uncertainty characterization (implicit in Step 8 model)

**Referenced Papers & Basis:**
- Prioritized sweeping: Peng & Williams (1993), Moore & Atkeson (1993)
- Small backups: Van Seijen & Sutton (2013) "Efficient Planning in MDPs with Small Backups"
- General planning: Sutton et al. (2008) "Dyna-style Planning"
- MCTS: Suggested reference (Monte Carlo Tree Search, Kocsis & Szepesvári)

---

### Step 10: Prototype-AI II — The STOMP Progression

**Exact Name & Description:**
> "Prototype-AI II: The STOMP progression. Now we introduce subtasks and temporal abstraction. The highest ranked features are made each into a separate reward-respecting subtask with a terminal value that encourages ending when the feature is high. Each subtask is solved to produce an option. For each such option, its model is learned and added to the transition model used for planning."

**STOMP = SubTask → Option → Model → Planning**

**Core Problem Statement:**
- Introduce **options** (Sutton, Precup, Singh 1999): temporally abstract actions
- Each option has:
  - **Policy π_o**: how to act to achieve the subtask
  - **Termination condition β_o**: when to stop
  - **Model M_o**: how environment changes under the option
- Options enable planning over longer horizons with fewer steps

**Key Algorithmic Requirements:**
1. **Subtask definition**
   - Highest-ranked features become subtask goals
   - Subtask: maximize cumulative feature value before termination
   - Reward-respecting: primary reward signal always counts
2. **Option learning**
   - **Off-policy learning**: Current (higher-level) policy may not follow option's own policy
   - Optimize option policy π_o to maximize feature while respecting primary reward
3. **Option model learning**
   - Learn **M_o(s, o): s → E[r, s_τ]** where τ is option termination time
   - Predicts cumulative reward and final state at option termination
4. **Multi-level planning**
   - Plan over options, not primitive actions
   - Option model enables long-horizon DP
5. **Feature-to-option mapping**
   - Feature ranking drives subtask selection
   - High-utility features → subtasks → options
   - Low-utility features → delete option → create new subtask

**Off-Policy Learning Requirement:**
> "The learning processes are conditional on the option, and so will need to be done off-policy."

The agent may interrupt option at any time (higher-level decision) or follow extrinsic reward better than staying in option.

**Success Criteria:**
> "[STOMP enables] temporally abstract cognitive structure."

Specific evidence targets:
- Options reduce effective horizon (faster planning)
- Option models learned accurately under off-policy data
- Feature-driven subtask discovery creates meaningful options
- Long-horizon planning improves over one-step Prototype-AI I

**The STOMP Cycle (Figure 3 in Paper):**
```
State Features → Subtasks (highest-ranked features)
              ↓
         Options (policies for subtasks)
              ↓
      Option Models (transition models)
              ↓
         Planning Process
              ↓
    Value functions & Policies (improved)
         ↓ ↑
     Learning Loop
```

**Robot/Real-World Requirements:**
- Options enable real-time hierarchical control
- Feature-driven abstraction allows automatic skill discovery
- Long-horizon planning without exponential branching
- Off-policy learning handles interruption/preemption

**Referenced Papers & Basis:**
- Options: Sutton, Precup, Singh (1999) "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction"
- Option models: Sutton et al. (2022) "Reward-Respecting Subtasks for Model-Based RL" (arXiv:2202.03466)
- STOMP (as framework): Sutton et al. (2022) same paper

---

### Step 11: Prototype-AI III — Oak Architecture

**Exact Name & Description:**
> "Prototype-AI III: Oak. The Oak architecture modifies Prototype-AI II by adding feedback processes that continually assess the utility of all the elements (features, subtasks, options, and option models) and determine which elements should be removed and replaced with new elements."

**Core Problem Statement:**
- Prototype-AI II has static elements (once learned, features/options persist)
- Oak adds **continuous lifecycle management**: assessment → replacement cycle
- Applies to **all** abstraction levels: features → subtasks → options → models

**Key Algorithmic Requirements:**
1. **Utility assessment of all elements**
   - **Features**: How much do they contribute to learning/planning?
   - **Subtasks**: Are their options being used/useful?
   - **Options**: Do their models improve planning? Do they advance the policy?
   - **Models**: Are they accurate? Do they improve value estimates?
2. **Continuous reranking**
   - Features reranked frequently (every step or per-episode)
   - High-utility features stay as subtask basis
   - Low-utility features → delete their option/subtask → free budget for new candidate
3. **Replacement strategy**
   - Identify which element to delete (weakest)
   - Generate new candidate (variant of deleted, or novel)
   - Test new candidate (Step 2 candidate testing logic)
   - Promote if better, discard if worse
4. **Option Keyboard (optional enhancement)**
   - Address: "How to combine options?"
   - Each option indexed by a one-hot vector (one per subtask)
   - **Chord options**: Play multiple keys simultaneously
     - Option keyboard vector = mixture of component options
     - Or: keyboard vector defines a multi-feature subtask (maximize multiple features)
   - **Two designs** for chord options:
     - **Design 1**: Fixed blend of component options; model learns chord policies
     - **Design 2**: Keyboard vector is interpreted as multi-feature subtask; options learned toward multi-feature reward

**Lifecycle Cycle (Figure 3 & Step 11 text):**
```
State & Time Abstractions (features, subtasks, options, models)
       ↓ (learning/planning feedback)
Continuous Utility Assessment
       ↓ (identify low-utility)
Element Deletion
       ↓ (generate candidate)
New Element Candidate
       ↓ (test & promote)
Feature/Option Lifecycle Update
       ↓ (feedback to all)
Reranking & Replacement
```

**Success Criteria:**
> "In these and other ways, the state and time abstractions are continually changing and improving."

Specific evidence targets:
- Automated discovery of useful features/options outperforms static-set baseline
- Replacement cycle finds good new options without manual design
- Deletion doesn't catastrophically harm performance (graceful degradation)
- Hierarchy scales: allows arbitrary-depth feature stacking

**Option Keyboard Enhancements:**
- Enable **skill composition**: combine learned options into new behaviors
- Reduce manual option engineering
- Natural multi-objective control (balance multiple subtasks)

**Robot/Real-World Requirements:**
- **Lifelong learning**: System adapts to non-stationary environments
- **Automatic curriculum**: Difficult features/options are discovered gradually
- **Resource efficiency**: Budget enforced; no unbounded growth
- **Graceful shutdown**: Utilities drop → options deleted → agent continues with remaining skills

**Referenced Papers & Basis:**
- Option keyboard: Barreto et al. (2019) "The Option Keyboard: Combining Skills in Reinforcement Learning"
- Subtasks & STOMP: Sutton et al. (2022)
- Feature lifecycle: Multi-head learner deletion (Step 2 extension)

---

### Step 12: Prototype-IA — Intelligence Amplification

**Exact Name & Description:**
> "Prototype-IA: Intelligence amplification. A demonstration of intelligence amplification (IA), wherein a Prototype-AI II agent is shown to increase the speed and overall decision-making capacity of a second agent in non-trivial ways."

**Core Problem Statement:**
- Different from Prototype-AI alone: **two agents interact**
- Primary agent (Prototype-IA) augments a second agent (human or AI) by:
  - Providing predictions (exo-cerebellum)
  - Providing action recommendations & policies (exo-cortex)
  - Learning what information the partner agent needs

**Two IA Implementations:**

#### Version 1: Computational Exo-Cerebellum
- Predicts outcomes using **GVF predictions** (Step 3 foundation)
- Provides perceptual signals (e.g., "ball trajectory," "collision prediction")
- Augments partner's sensory apparatus
- Based on prediction and feature construction (Steps 2-3)

#### Version 2: Computational Exo-Cortex
- Full policy and planning augmentation
- Uses **options & hierarchical planning** (Steps 10-11)
- Recommends actions or long-horizon plans
- Learns to rank recommendations by partner's actual preferences
- Fully manifests "multiplicative enhancement" of intelligence

**Key Algorithmic Requirements:**
1. **Dual-agent learning**
   - Primary agent observes partner's actions and outcomes
   - Learns what features/predictions the partner values
   - Learns when to signal/recommend
2. **Communication bandwidth management**
   - Don't overload partner with information
   - Select high-value signals/recommendations
   - Adapt to partner's attention/processing capacity
3. **Online adaptation to partner**
   - Partner's utilities/preferences may be unknown initially
   - Learn model of partner: **What does this agent want?**
   - Continually refine recommendations
4. **Theoretical foundation: Intelligence Amplification**
   - Historical: Ashby (1956), Licklider (1960), Engelbart (1962)
   - Agent as **cognitive prosthesis**: extends partner's decision speed and scope

**Success Criteria:**
> "A demonstration of intelligence amplification (IA), wherein a Prototype-AI II agent is shown to increase the speed and overall decision-making capacity of a second agent in non-trivial ways."

Specific evidence targets:
- Partner's decision time decreases with IA system (speed)
- Partner makes better decisions (accuracy) with IA recommendations
- IA learning is continual: recommendations improve over time
- Works in both human-agent and agent-agent settings

**Agent-Agent Setting Example (Pilarski et al. 2022):**
- GVF co-agent signals valuable state estimates to decision-making agent
- Coordination learned without explicit communication protocol
- Both agents improve through interaction

**Human-Agent Setting (Conceptual):**
- Robotic arm augments human surgeon with tremor filtering + trajectory prediction
- Displays help surgeon perceive hidden anatomy via GVF predictions
- System learns surgeon's hand-motion preferences and infers intent

**Robot/Real-World Requirements:**
- **Real-time coordination**: IA must respond in <100ms
- **Graceful degradation**: System works even if IA is slow/delayed
- **Human interpretability** (for human-agent IA): Recommendations must be understandable
- **Learning from human feedback**: System improves as humans correct it

**Referenced Papers & Basis:**
- Intelligence Amplification history: Ashby (1956) "Introduction to Cybernetics", Licklider (1960) "Man-Computer Symbiosis", Engelbart (1962) "Augmenting Human Intellect"
- Example application: Pilarski et al. (2022) "The Frost Hollow Experiments: Pavlovian Signalling as a Path to Coordination and Communication between Agents" (arXiv:2203.09498)
- GVF prediction foundation: Steps 1-3
- Hierarchical action recommendation: Steps 10-11 (Oak options)

---

## Summary Table: 12 Steps at a Glance

| Step | Name | Domain | Key Innovation | Foundation |
|------|------|--------|-----------------|-----------|
| 1 | Representation I | Supervised (fixed features) | Per-feature adaptive step-sizes + online normalization | IDBD, Autostep |
| 2 | Representation II | Supervised (feature finding) | Smart feature generation + resource budgeting | Step 1 + feature DAG |
| 3 | Prediction I | GVF prediction | Multi-demon prediction + traces | Steps 1-2 + GVF |
| 4 | Control I | Actor-critic (progressing complexity) | Critic from Steps 1-3 + policy learning | Actor-critic + Steps 1-3 |
| 5 | Prediction II | Average-reward GVF | Extend Step 3 to continuing problems | Step 3 + average-reward DP |
| 6 | Control II | Continuing control benchmarks | Adapt episodic benchmarks to continuing | Step 5 + benchmark suite |
| 7 | Planning I | Average-reward DP | Asynchronous DP + function approximation | Bertsekas + Sutton et al. |
| 8 | Prototype-AI I | One-step model-based RL | Integrated agent: perception + model + planning | Steps 1-7 |
| 9 | Planning II | Search control | Prioritized sweeping with function approx. | Peng & Williams + Van Seijen |
| 10 | Prototype-AI II | Temporal abstraction | STOMP: features → subtasks → options → models | Steps 8-9 + Sutton et al. 2022 |
| 11 | Prototype-AI III | Lifelong abstraction management | Oak: continuous utility assessment + replacement | Step 10 + feature lifecycle |
| 12 | Prototype-IA | Intelligence amplification | Dual-agent augmentation (exo-cerebellum/cortex) | GVF + hierarchical control |

---

## Architectural Vision: The Complete System

### High-Level System Diagram (from Paper Figure 2)

```
                  Agent
        [Foreground Processes - Every Time Step]
        
Observation → Perception (state construction)
                    ↓ (state)
              Reactive Policies (action selection)
                    ↓ (action)
              Action → Environment
              
              Reward ← Environment
                    ↓
            Value Functions (evaluate state/action)
                    ↓ (learning signals)
            
            [Learning processes continual, per-step]
            
        [Background Processes - Asynchronous/Parallel]
        
            Transition Model (world dynamics)
                    ↓
                Planning (asynchronous DP)
                    ↓ (value/policy improvement)
        
        [All components update every time step or continuously]
```

### Key Principles

1. **Temporal Uniformity**: No special training phases. Learning, planning, feature discovery, abstraction management all happen at every time step or continuously in the background.

2. **Continual Operation**: No episodic resets (unless task-specific). Agent runs indefinitely; algorithms must handle non-stationarity, catastrophic forgetting, and knowledge retention.

3. **Computational Realism**: Prioritize methods that scale with compute (search, learning). Must respect latency constraints (real-time control loop: <100ms).

4. **Experiential Grounding**: Only observations, actions, rewards. No hand-designed state, no human labels, no access to environment internals.

5. **Hierarchical Abstraction**: States → Features → Subtasks → Options → Models → Planning. Each level is learned and continually refined.

### Differences from Standard RL

| Aspect | Standard RL | Alberta Plan |
|--------|------------|--------------|
| **Horizon** | Episodic (finite, resets) | Continuing (infinite, no resets) |
| **Reward Signal** | Single, global | Average reward + GVF cumulants |
| **Learning Phase** | Offline batch or episodic | Continual, every time step |
| **Feature Representation** | Fixed (CNN) or learned offline (pre-training) | Continually constructed & replaced |
| **Planning** | Optional, offline (AlphaGo) | Continuous, asynchronous, foreground |
| **Abstraction** | Manual (hierarchical RL) | Learned from feature ranking (STOMP/Oak) |
| **Meta-Learning** | Outer loop (separate from agent) | Integrated (per-step step-size, feature selection) |
| **Deployment** | Sim-to-real, hand-tuning | Daemon-ready: drop into real-time control loop |

### Example: Robotic Manipulation (Conceptual)

**Step 1-3**: Learn to predict force/torque feedback from proprioception (linear prediction → nonlinear features → GVF multi-step prediction)

**Step 4-6**: Learn control policies (actor-critic) in simulated or real continuing manipulation task

**Step 7-8**: Build one-step model of object dynamics; plan to improve grasp stability

**Step 9**: Prioritize high-uncertainty states during planning (e.g., contact transitions)

**Step 10-11**: Discover subtasks (e.g., "move to stable grasp," "lift object," "place gently") as options; delete unused options; keep high-utility ones

**Step 12**: Augment human operator with predicted forces (exo-cerebellum) and action recommendations (exo-cortex); human can override, system learns human preferences

---

## Success Criteria Summary

### What "Done" Means for Each Step

| Step | Done When... |
|------|-------------|
| 1 | Tuning-free adaptive step-sizes beat hand-tuned LMS on sparse-relevance, drift, and scale-shift tasks |
| 2 | Feature-construction methods beat fair MLPs on out-of-hypothesis-class synthetic tasks without offline training |
| 3 | GVF prediction with learned features works on sequential tasks; off-policy learning converges |
| 4 | Actor-critic learns continually in bandit → contextual bandit → sequential control |
| 5 | Average-reward GVF learning extends Step 3 with minimal algorithmic changes |
| 6 | Continuing benchmark suite exists and distinguishes algorithms |
| 7 | Asynchronous DP with function approximation converges to ε-optimal average-reward policies |
| 8 | Integrated one-step model-based agent runs without divergence; outperforms model-free |
| 9 | Prioritized planning order improves convergence speed over random planning |
| 10 | STOMP discovers meaningful options; off-policy learning handles option interruption |
| 11 | Oak lifecycle management deletes weak options and promotes strong ones without collapse |
| 12 | IA system demonstrably speeds/improves partner agent's decision-making in non-trivial task |

---

## References Cited in Paper

**Core Methods:**
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.)
- Sutton, R. S., Precup, D., & Singh, S. (1999). "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction"
- Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., & Precup, D. (2011). "Horde: A Scalable Real-Time Architecture"

**Step 1 Methods:**
- Sutton (1992) IDBD / Adapting Bias by Gradient Descent
- Mahmood et al. (2012) Autostep / Tuning-Free Step-Size Adaptation
- Kearney et al. (2022) Autostep-for-GTD(λ)
- Degris (in prep.) Auto [unpublished]

**Step 2 & Feature Learning:**
- Sutton et al. (2022) "Reward-Respecting Subtasks for Model-Based RL"

**Planning & Search:**
- Peng & Williams (1993) Prioritized Sweeping
- Moore & Atkeson (1993) Prioritized Sweeping
- Sutton et al. (2008) Dyna-style Planning with Linear Function Approximation
- Van Seijen & Sutton (2013) Small Backups
- McMahan & Gordon (2005) Fast Exact Planning in MDPs

**Temporal Abstraction:**
- Sutton, Machado, Holland et al. (2022) Reward-Respecting Subtasks

**Intelligence Amplification:**
- Ashby, W. R. (1956) Introduction to Cybernetics
- Licklider, J. C. R. (1960) Man-Computer Symbiosis
- Engelbart, D. C. (1962) Augmenting Human Intellect
- Barreto et al. (2019) Option Keyboard
- Pilarski et al. (2022) Frost Hollow Experiments

---

## Gaps & Provisional Notes from Authors

The paper emphasizes that **this is a research roadmap, not a fixed plan**:

> "All research plans are suspect and provisional. Nevertheless, we must make them in order to communicate among ourselves and collaborate efficiently."

**Explicitly Uncertain Steps:**
- **Steps 10-12** are "less concrete and will probably evolve a lot as we approach them"
- **Subtask definition** (Step 10) remains open: how to automatically discover which features should become subtasks?
- **Search control** (Step 9) may not be "fully satisfactory" before temporal abstraction
- **Perception** (state construction from raw observations) is "perhaps the least well understood"

**Parallel Work Not in Roadmap:**
- Robotics applications (will interact with Steps 1-11)
- Intelligence Amplification (treated here in Step 12, but may need its own separate roadmap)
- Generalization to multi-agent coordination (mentioned for Step 12 but not explored)

**Scope Limits:**
- Plan focuses on **core learning/planning/control algorithms**
- Does not address: curriculum learning, meta-learning across tasks, continual transfer learning
- Assumes simulator or real-time environment access (not true offline/batch RL)

---

## How Existing Framework Implementation Maps to Steps

**(From CLAUDE.md & Assessment Docs)**

### Implemented
- **Step 1**: Complete (IDBD, Autostep, Adam, RMSprop + online normalization)
- **Step 2**: Partial (MLP + feature discovery, but not fully beating MLPs on all external tasks)
- **Step 3**: Partial (GVF prediction + Horde, but off-policy needs work)
- **Step 4**: Partial (SARSA control via Horde, but only on-policy so far)

### Primitive/Scaffolding
- **Step 5-6**: Minimal (average-reward DP exists, but no full integration)
- **Step 7**: Minimal (planning skeleton, not full asynchronous DP)
- **Step 8**: Primitive (one-step Dyna, but not integrated with Step 2 feature finding)
- **Step 9**: Primitive (planning interface, no search control)
- **Step 10-11**: Primitive (STOMP skeleton, option keyboard concept, lifecycle interfaces)
- **Step 12**: Primitive (GVF prediction base, but no dual-agent coordination)

---

## Conclusion

The Alberta Plan is a **comprehensive, honest roadmap** to building continual, model-based AI agents that learn from experience alone. It progresses from the simplest setting (linear supervised learning) to a complete hierarchical system (Oak) that discovers and manages its own abstractions, and finally to agents that can amplify the intelligence of partners.

The plan's distinguishing features are:
1. **Temporal uniformity**: Eliminates special training phases
2. **Computational realism**: Emphasizes scalable learning and planning
3. **Experiential grounding**: No hand-designed features or human labels
4. **Hierarchical learning**: Automatic abstraction discovery
5. **Modular progression**: Each step builds on prior work; can attach at any level

The implementation is advancing but still incomplete. Steps 1-2 are closest to the paper's vision; Steps 3-4 are functional but off-policy needs work; Steps 5-8 are prototyped; Steps 9-12 are scaffolded but need full integration.
