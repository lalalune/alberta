"""Alberta Framework: A JAX-based research framework for continual AI.

The Alberta Framework provides foundational components for continual reinforcement
learning research. Built on JAX for hardware acceleration, the framework emphasizes
temporal uniformity — every component updates at every time step, with no special
training phases or batch processing.

Roadmap
-------
| Step | Focus | Status |
|------|-------|--------|
| 1 | Meta-learned step-sizes (IDBD, Autostep) | **Complete** |
| 2 | Nonlinear function approximation (MLP, ObGD) | **In Progress** |
| 3 | GVF predictions, Horde architecture | Planned |
| 4 | Actor-critic with eligibility traces | Planned |
| 5-6 | Off-policy learning, average reward | Planned |
| 7-12 | Hierarchical, multi-agent, world models | Future |

Examples
--------
```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop

# Non-stationary stream where target weights drift over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Learner with IDBD meta-learned step-sizes
learner = LinearLearner(optimizer=IDBD())

# JIT-compiled training via jax.lax.scan
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

References
----------
- The Alberta Plan for AI Research (Sutton et al., 2022): https://arxiv.org/abs/2208.11173
- Adapting Bias by Gradient Descent (Sutton, 1992)
- Tuning-free Step-size Adaptation (Mahmood et al., 2012)
- Streaming Deep Reinforcement Learning Finally Works (Elsayed et al., 2024)
"""

__version__ = "0.17.1"

# Baseline optimizers
from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ActorCriticState,
    ActorCriticUpdateResult,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    ContinuousActorCriticState,
    ContinuousActorCriticUpdateResult,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.baseline_optimizers import (
    NADALINE,
    AdaGain,
    AdaGainState,
    Adam,
    AdamParamState,
    AdamState,
    NadalineState,
    RMSprop,
    RMSpropParamState,
    RMSpropState,
)

# Checkpoint utilities
from alberta_framework.core.checkpoints import (
    checkpoint_exists,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.continual_backprop import (
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
)
from alberta_framework.core.cumulant_discovery import CumulantDiscovery

# Diagnostics
from alberta_framework.core.diagnostics import (
    FeatureRelevance,
    compute_feature_relevance,
    compute_feature_sensitivity,
    relevance_to_dict,
)
from alberta_framework.core.diffeml import (
    BOOLEAN_INPUTS,
    DiffEMLGateSelector,
    DiffEMLLearner,
    EMLTreeLearner,
    EMLTreeState,
    boolean_truth_table,
    build_eml_template_bank,
    eml_operator,
    eml_threshold_gate_library,
    evaluate_eml_template_bank,
    mask_from_truth_table,
    run_diffeml_learning_loop,
    run_eml_tree_learning_loop,
    stable_eml_operator,
)
from alberta_framework.core.feature_discovery import (
    FixedBudgetFeatureLearner,
    run_feature_discovery_arrays,
    run_feature_discovery_loop,
)
from alberta_framework.core.history_features import HistoryFeatureExtractor

# Horde / GVF (Step 3)
from alberta_framework.core.horde import (
    BatchedHordeResult,
    HordeLearner,
    HordeLearningResult,
    HordeUpdateResult,
    MixedHorde,
    MixedHordeLearningResult,
    MixedHordeState,
    run_horde_learning_loop,
    run_horde_learning_loop_batched,
    run_horde_learning_loop_final_state,
    run_mixed_horde_learning_loop,
)

# Horde Actor-Critic (Step 4)
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticArrayResult,
    HordeActorCriticConfig,
    HordeActorCriticState,
    HordeActorCriticUpdateResult,
    run_horde_actor_critic_from_arrays,
)

# Core types
# Learners
# Initializers
from alberta_framework.core.initializers import sparse_init
from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    run_interaction_feature_arrays,
)
from alberta_framework.core.learners import (
    LinearLearner,
    MLPLearner,
    MLPUpdateResult,
    TDLinearLearner,
    TDUpdateResult,
    TrueOnlineTDLearner,
    TrueOnlineTDState,
    TrueOnlineTDUpdateResult,
    UpdateResult,
    metrics_to_dicts,
    run_learning_loop,
    run_learning_loop_batched,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_td_learning_loop,
    run_true_online_td_loop,
)

# Multi-head learner
from alberta_framework.core.multi_head_learner import (
    BatchedMultiHeadResult,
    MultiHeadLearningResult,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
    multi_head_metrics_to_dicts,
    run_multi_head_learning_loop,
    run_multi_head_learning_loop_batched,
)

# Normalizers
from alberta_framework.core.normalizers import (
    AnyNormalizerState,
    EMANormalizer,
    EMANormalizerState,
    Normalizer,
    StreamingBatchNormalizer,
    StreamingBatchNormalizerState,
    WelfordNormalizer,
    WelfordNormalizerState,
    normalizer_from_config,
)
from alberta_framework.core.off_policy_td import OffPolicyTDLinearLearner

# Optimizers
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    TDIDBD,
    AGCBounding,
    Autostep,
    AutostepGTDLambda,
    AutoTDIDBD,
    Bounder,
    ObGD,
    ObGDBounding,
    Optimizer,
    TDOptimizer,
    TDOptimizerUpdate,
    bounder_from_config,
    optimizer_from_config,
)
from alberta_framework.core.resource_manager import (
    GeneratorMetaResourceManager,
    LearnedResourceManager,
)

# SARSA (Step 4a)
from alberta_framework.core.sarsa import (
    SARSAAgent,
    SARSAArrayResult,
    SARSAConfig,
    SARSAContinuingResult,
    SARSAEpisodeResult,
    SARSAState,
    SARSAUpdateResult,
    run_sarsa_continuing,
    run_sarsa_episode,
    run_sarsa_from_arrays,
    run_sarsa_from_arrays_final_state,
)

# UPGD (Step 2)
from alberta_framework.core.upgd import (
    UPGDLearner,
    UPGDLearningResult,
    UPGDState,
    UPGDUpdateResult,
    run_upgd_arrays,
    run_upgd_loop,
)
from alberta_framework.security import (
    N_SECURITY_ACTIONS,
    SECURITY_ACTION_NAMES,
    SECURITY_GYM_ACTION_NAMES,
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRewardWeights,
    SecurityRolloutStep,
    ThroughputMeter,
    coerce_security_action,
    security_gym_action_name,
    security_gym_action_reward,
    security_reward,
    to_security_gym_action,
    validate_security_rollout,
)
from alberta_framework.steps.step2 import (
    Step2HybridConfig,
    make_step2_hybrid_learner,
)

# Production Step 1-4 pipeline. Keep this optional at package-import time so
# core Step 1/2 learners and research scripts remain usable while pipeline
# dependencies are under active development.
try:
    from alberta_framework.pipeline import (
        AlbertaPipeline,
        AlbertaPipelineArrayResult,
        AlbertaPipelineConfig,
        AlbertaPipelineSmokeResult,
        AlbertaPipelineState,
        AlbertaPipelineStepResult,
        ControlMode,
        CumulantFn,
        HordeActorCriticPipelineConfig,
        Step2FeatureConfig,
        Step2Mode,
        Step2UPGDConfig,
        make_alberta_pipeline,
        run_pipeline_smoke,
    )

    _pipeline_available = True
except ImportError:
    _pipeline_available = False
from alberta_framework.core.types import (
    AutostepGTDLambdaState,
    AutostepParamState,
    AutostepState,
    AutoTDIDBDState,
    BatchedLearningResult,
    BatchedMLPResult,
    DemonType,
    GVFSpec,
    HordeSpec,
    IDBDParamState,
    IDBDState,
    LearnerState,
    LMSState,
    MLPLearnerState,
    MLPParams,
    NormalizerHistory,
    NormalizerTrackingConfig,
    ObGDState,
    Observation,
    Prediction,
    StepSizeHistory,
    StepSizeTrackingConfig,
    Target,
    TDIDBDState,
    TDLearnerState,
    TDTimeStep,
    TimeStep,
    TraceMode,
    agent_age_s,
    agent_uptime_s,
    create_autotdidbd_state,
    create_horde_spec,
    create_obgd_state,
    create_tdidbd_state,
)

# Streams - base
from alberta_framework.streams.base import ScanStream

# Streams - Step 2 feature discovery
from alberta_framework.streams.feature_discovery import (
    InteractionFeatureDiscoveryState,
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryState,
    NonlinearFeatureDiscoveryStream,
    collect_feature_discovery_stream,
)
from alberta_framework.streams.out_of_class import (
    CompositionalState,
    CompositionalStream,
    FrequencyMismatchState,
    FrequencyMismatchStream,
    OutOfClassPolynomialState,
    OutOfClassPolynomialStream,
)
from alberta_framework.streams.partial_observation import MaskMode, PartialObservationWrapper

# Streams - synthetic
from alberta_framework.streams.synthetic import (
    AbruptChangeState,
    AbruptChangeStream,
    AbruptChangeTarget,
    CyclicState,
    CyclicStream,
    CyclicTarget,
    DynamicScaleShiftState,
    DynamicScaleShiftStream,
    HiddenStateAR2State,
    HiddenStateAR2Stream,
    PeriodicChangeState,
    PeriodicChangeStream,
    PeriodicChangeTarget,
    RandomWalkState,
    RandomWalkStream,
    RandomWalkTarget,
    ScaleDriftState,
    ScaleDriftStream,
    ScaledStreamState,
    ScaledStreamWrapper,
    SuttonExperiment1State,
    SuttonExperiment1Stream,
    make_scale_range,
)

# Utilities
from alberta_framework.utils.metrics import (
    compare_learners,
    compute_cumulative_error,
    compute_running_mean,
    compute_tracking_error,
    extract_metric,
)
from alberta_framework.utils.nexting import multi_channel_horizon_returns
from alberta_framework.utils.timing import Timer, format_duration

# Gymnasium streams (optional)
try:
    from alberta_framework.streams.gymnasium import (
        GymnasiumStream,
        PredictionMode,
        TDStream,
        collect_trajectory,
        learn_from_trajectory,
        learn_from_trajectory_normalized,
        make_epsilon_greedy_policy,
        make_gymnasium_stream,
        make_random_policy,
    )

    _gymnasium_available = True
except ImportError:
    _gymnasium_available = False

__all__ = [
    # Version
    "__version__",
    # Types - Supervised Learning
    "AutostepParamState",
    "AutostepGTDLambdaState",
    "AutostepState",
    "BatchedLearningResult",
    "IDBDParamState",
    "IDBDState",
    "LMSState",
    "LearnerState",
    "NormalizerHistory",
    "AnyNormalizerState",
    "EMANormalizerState",
    "StreamingBatchNormalizerState",
    "WelfordNormalizerState",
    "NormalizerTrackingConfig",
    "ObGDState",
    "Observation",
    "Prediction",
    "StepSizeHistory",
    "StepSizeTrackingConfig",
    "Target",
    "TimeStep",
    "TraceMode",
    "UpdateResult",
    # Types - MLP
    "BatchedMLPResult",
    "MLPLearnerState",
    "MLPParams",
    "MLPUpdateResult",
    # Types - TD Learning
    "AutoTDIDBDState",
    "TDIDBDState",
    "TDLearnerState",
    "TDTimeStep",
    "TDUpdateResult",
    "TrueOnlineTDState",
    "TrueOnlineTDUpdateResult",
    "agent_age_s",
    "agent_uptime_s",
    "create_obgd_state",
    "create_tdidbd_state",
    "create_autotdidbd_state",
    # Baseline optimizers
    "AdaGain",
    "AdaGainState",
    "Adam",
    "AdamParamState",
    "AdamState",
    "NADALINE",
    "NadalineState",
    "RMSprop",
    "RMSpropParamState",
    "RMSpropState",
    # Optimizers - Supervised Learning
    "AGCBounding",
    "Autostep",
    "AutostepGTDLambda",
    "Bounder",
    "IDBD",
    "LMS",
    "ObGD",
    "ObGDBounding",
    "Optimizer",
    "optimizer_from_config",
    "bounder_from_config",
    # Optimizers - TD Learning
    "AutoTDIDBD",
    "TDIDBD",
    "TDOptimizer",
    "TDOptimizerUpdate",
    # Initializers
    "sparse_init",
    # Normalizers
    "Normalizer",
    "EMANormalizer",
    "StreamingBatchNormalizer",
    "WelfordNormalizer",
    "normalizer_from_config",
    # Learners - Supervised Learning
    "LinearLearner",
    "run_learning_loop",
    "run_learning_loop_batched",
    "metrics_to_dicts",
    # Learners - MLP
    "MLPLearner",
    "run_mlp_learning_loop",
    "run_mlp_learning_loop_batched",
    # Learners - Multi-Head MLP
    "BatchedMultiHeadResult",
    "MultiHeadLearningResult",
    "MultiHeadMLPLearner",
    "MultiHeadMLPState",
    "MultiHeadMLPUpdateResult",
    "multi_head_metrics_to_dicts",
    "run_multi_head_learning_loop",
    "run_multi_head_learning_loop_batched",
    # Learners - UPGD (Step 2)
    "CumulantDiscovery",
    "FixedBudgetFeatureLearner",
    "FixedBudgetInteractionLearner",
    "GeneratorMetaResourceManager",
    "HistoryFeatureExtractor",
    "LearnedResourceManager",
    "OffPolicyTDLinearLearner",
    "Step2HybridConfig",
    "CBPMultiHeadMLPLearner",
    "ContinualBackpropConfig",
    "UPGDLearner",
    "UPGDLearningResult",
    "UPGDState",
    "UPGDUpdateResult",
    "make_step2_hybrid_learner",
    "run_feature_discovery_arrays",
    "run_feature_discovery_loop",
    "run_interaction_feature_arrays",
    "run_upgd_arrays",
    "run_upgd_loop",
    # Differentiable EML
    "BOOLEAN_INPUTS",
    "DiffEMLGateSelector",
    "DiffEMLLearner",
    "EMLTreeLearner",
    "EMLTreeState",
    "boolean_truth_table",
    "build_eml_template_bank",
    "eml_operator",
    "eml_threshold_gate_library",
    "evaluate_eml_template_bank",
    "mask_from_truth_table",
    "run_diffeml_learning_loop",
    "run_eml_tree_learning_loop",
    "stable_eml_operator",
    # GVF / Horde (Step 3)
    "BatchedHordeResult",
    "DemonType",
    "GVFSpec",
    "HordeLearner",
    "HordeLearningResult",
    "HordeSpec",
    "HordeUpdateResult",
    "MixedHorde",
    "MixedHordeLearningResult",
    "MixedHordeState",
    "create_horde_spec",
    "run_horde_learning_loop",
    "run_horde_learning_loop_batched",
    "run_horde_learning_loop_final_state",
    "run_mixed_horde_learning_loop",
    # Horde Actor-Critic (Step 4)
    "HordeActorCriticAgent",
    "HordeActorCriticArrayResult",
    "HordeActorCriticConfig",
    "HordeActorCriticState",
    "HordeActorCriticUpdateResult",
    "run_horde_actor_critic_from_arrays",
    # Actor-Critic (Step 4)
    "ActorCriticAgent",
    "ActorCriticConfig",
    "ActorCriticState",
    "ActorCriticUpdateResult",
    "ContinuousActorCriticAgent",
    "ContinuousActorCriticConfig",
    "ContinuousActorCriticState",
    "ContinuousActorCriticUpdateResult",
    "run_actor_critic_from_arrays",
    "run_continuous_actor_critic_from_arrays",
    # SARSA (Step 4a)
    "SARSAAgent",
    "SARSAArrayResult",
    "SARSAConfig",
    "SARSAContinuingResult",
    "SARSAEpisodeResult",
    "SARSAState",
    "SARSAUpdateResult",
    "run_sarsa_continuing",
    "run_sarsa_episode",
    "run_sarsa_from_arrays",
    "run_sarsa_from_arrays_final_state",
    # Learners - TD Learning
    "TDLinearLearner",
    "TrueOnlineTDLearner",
    "run_td_learning_loop",
    "run_true_online_td_loop",
    # Streams - protocol
    "ScanStream",
    # Streams - Step 2 feature discovery
    "InteractionFeatureDiscoveryState",
    "InteractionFeatureDiscoveryStream",
    "NonlinearFeatureDiscoveryState",
    "NonlinearFeatureDiscoveryStream",
    "collect_feature_discovery_stream",
    # Streams - Step 2 out-of-class
    "CompositionalState",
    "CompositionalStream",
    "FrequencyMismatchState",
    "FrequencyMismatchStream",
    "OutOfClassPolynomialState",
    "OutOfClassPolynomialStream",
    "MaskMode",
    "PartialObservationWrapper",
    # Streams - synthetic
    "AbruptChangeState",
    "AbruptChangeStream",
    "AbruptChangeTarget",
    "CyclicState",
    "CyclicStream",
    "CyclicTarget",
    "DynamicScaleShiftState",
    "DynamicScaleShiftStream",
    "PeriodicChangeState",
    "PeriodicChangeStream",
    "PeriodicChangeTarget",
    "RandomWalkState",
    "RandomWalkStream",
    "RandomWalkTarget",
    "HiddenStateAR2State",
    "HiddenStateAR2Stream",
    "ScaleDriftState",
    "ScaleDriftStream",
    "ScaledStreamState",
    "ScaledStreamWrapper",
    "SuttonExperiment1State",
    "SuttonExperiment1Stream",
    # Stream utilities
    "make_scale_range",
    # Utilities
    "compare_learners",
    "compute_cumulative_error",
    "compute_running_mean",
    "compute_tracking_error",
    "extract_metric",
    "multi_channel_horizon_returns",
    # Checkpoint utilities
    "checkpoint_exists",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "save_checkpoint",
    # Diagnostics
    "FeatureRelevance",
    "compute_feature_relevance",
    "compute_feature_sensitivity",
    "relevance_to_dict",
    # Security integration
    "N_SECURITY_ACTIONS",
    "SECURITY_ACTION_NAMES",
    "SECURITY_GYM_ACTION_NAMES",
    "SecurityAction",
    "SecurityFeatureSchema",
    "SecurityRewardWeights",
    "SecurityRolloutStep",
    "ThroughputMeter",
    "coerce_security_action",
    "security_gym_action_name",
    "security_gym_action_reward",
    "security_reward",
    "to_security_gym_action",
    "validate_security_rollout",
    # Timing
    "Timer",
    "format_duration",
]

# Add Gymnasium exports if available
if _gymnasium_available:
    __all__ += [
        "GymnasiumStream",
        "PredictionMode",
        "TDStream",
        "collect_trajectory",
        "learn_from_trajectory",
        "learn_from_trajectory_normalized",
        "make_epsilon_greedy_policy",
        "make_gymnasium_stream",
        "make_random_policy",
    ]

if _pipeline_available:
    __all__ += [
        "AlbertaPipeline",
        "AlbertaPipelineArrayResult",
        "AlbertaPipelineConfig",
        "AlbertaPipelineSmokeResult",
        "AlbertaPipelineState",
        "AlbertaPipelineStepResult",
        "ControlMode",
        "CumulantFn",
        "HordeActorCriticPipelineConfig",
        "Step2FeatureConfig",
        "Step2Mode",
        "Step2UPGDConfig",
        "make_alberta_pipeline",
        "run_pipeline_smoke",
    ]
