# MedRLVR — Step-Verifiable Reinforcement Learning for Medical Tool-Using Agents

> *Verification granularity, not verifier strength, is the bottleneck for safe deployment of medical RL agents.*

---

## Motivation

Medical LLM evaluation is moving past USMLE-style multiple choice toward
structured, multi-step tool use: dosage calculation, ICD coding,
risk-stratification scoring (CHA₂DS₂-VASc, MELD, SOFA, ...), guideline
retrieval. Each step in such a trajectory corresponds to an explicit clinical
decision point — and each is also a place where the model can fail in a way
that should be detectable.

Two facts shape the design space:

1. **Supervised fine-tuning has plateaued for medical agentic reasoning.**
   High-quality clinical trajectory annotation is expensive, and SFT tends to
   learn surface form rather than the underlying reasoning process.
2. **Reinforcement learning with verifiable rewards (RLVR)** — as demonstrated
   by DeepSeek-R1, Tulu-3, and ToolRL on math and code — sidesteps the
   annotation bottleneck: as long as a programmatic verifier exists, a policy
   can learn from its own rollouts. This is exactly what is required when a
   research setting offers neither IRB access nor clinical collaborators.

The catch: most RLVR work to date relies on a **single outcome-level reward**.
In a medical multi-step task an error may originate at step 1 (wrong tool) or
step 4 (misread output), and an outcome reward cannot disentangle the two —
the classical credit-assignment problem, now with clinical stakes.

---

## Core Hypothesis

> In multi-step compositional medical tasks, **step-level verifiable rewards**
> systematically outperform outcome-level rewards in **sample efficiency**,
> **cross-tool generalization**, and **robustness to verifier-quality
> degradation**.

This is the falsifiable scientific claim the project is built around.

---

## Key Insights

1. **Granularity beats fidelity.** A clean outcome verifier is not enough —
   what matters is *where in the trajectory the verifier can speak*.
   Step-level rewards offer a "voting" effect: even when individual signals
   are noisy, their diversity stabilizes credit assignment.

2. **Computation is the cleanest signal in medicine.** Clinical calculators
   are deterministic Python; their outputs can be re-executed and checked
   exactly. This makes the compute reward the only sub-signal *guaranteed
   reliable* at deployment time, and motivates treating it as a permanent
   anchor while other verifiers are allowed to degrade.

3. **Reward hacking is unacceptable in clinical settings.** Hacks must be
   detectable and quantifiable. A step-level decomposition makes specific
   failure modes — e.g. *right answer with wrong arguments* — directly
   observable rather than hidden inside a single scalar.

4. **Verifier degradation is a science question, not an engineering
   nuisance.** Real clinical verifiers are imperfect; characterising
   *when and how* RLVR breaks under noisy verification is the contribution
   most resistant to a "this is just a domain application" criticism.

---

## Novelty

Three differentiators against the closest related work:

| | Closest neighbour | Our contribution |
|---|---|---|
| **Domain** | Medical multi-agent prompting (MedAgents, MDAgents) | First systematic *training* framework for medical tool-using agents under RLVR |
| **Method** | Generic RLVR with outcome reward (R1, Tulu-3); PRM-based step rewards (Math-Shepherd) | A four-tier *rule-based* reward decomposition achieving step-level credit assignment **without** learning a process-reward model |
| **Science** | Generic ablations on reward shape | Explicit study of **verifier degradation** — how policy quality and reward hacking evolve as verification fidelity decays from oracle to 50 % flip noise |

---

## Method Overview

### Trajectory format

```
case x ─┬─→ step_1: (tool_1, args_1) → output_1 → reasoning_1
        ├─→ step_2: (tool_2, args_2) → output_2 → reasoning_2
        ├─→ ...
        └─→ step_T: final_answer y
```

A ReAct + JSON tool-call protocol is used so trajectories are mechanically
parseable:

```
<think>...reasoning...</think>
<tool_call>{"name": "CHA2DS2_VASc", "arguments": {"age": 72, "sex": "F", ...}}</tool_call>
<tool_response>{"score": 4, "interpretation": "..."}</tool_response>
<think>...</think>
<answer>4</answer>
```

Malformed trajectories receive zero on **every** sub-reward, preventing
format-evasion gaming.

### Four verifiable sub-rewards

| Sub-reward | Verification mechanism | Reliability |
|---|---|---|
| `r_select`  | Tool choice vs. gold tool set: 0.5·Jaccard + 0.5·LCS | High |
| `r_args`    | Per-key, type-aware match (numeric tolerance, boolean exact, enum normalized); LLM-judge fallback for free text | Medium-high |
| `r_compute` | Re-execute the called tool with gold arguments; exact match on output | **Perfect** (deterministic) |
| `r_final`   | Closed-form numeric match (MedCalc) or LLM-judge (open-ended) + format component | Medium-high |

Total reward: `R = α₁·r_select + α₂·r_args + α₃·r_compute + α₄·r_final`.
Four preset α-vectors define the experimental axis:

| Preset | α₁ | α₂ | α₃ | α₄ | Role |
|---|---|---|---|---|---|
| outcome           | 0    | 0    | 0    | 1.00 | Classical RLVR baseline |
| step_only         | 0.25 | 0.25 | 0.25 | 0.25 | Equal-weight ablation |
| **shaped** ★      | 0.15 | 0.20 | 0.30 | 0.35 | Default — heavier compute, anchored final |
| compute_anchored  | 0.05 | 0.10 | 0.50 | 0.35 | Extreme compute weighting |

### Verifier degradation

Five graded verifiers, built by injecting *p*-flip noise into otherwise
rule-based signals:

| Level | r_select | r_args | r_compute | r_final |
|---|---|---|---|---|
| V0 (oracle)       | rule       | rule       | rule | rule       |
| V1 (LLM judge)    | rule       | rule       | rule | LLM judge  |
| V2 (10 % noise)   | flip(0.1)  | flip(0.1)  | rule | flip(0.1)  |
| V3 (30 % noise)   | flip(0.3)  | flip(0.3)  | rule | flip(0.3)  |
| V4 (50 % noise)   | flip(0.5)  | flip(0.5)  | rule | flip(0.5)  |

`r_compute` is **never** degraded — clinical calculators are deterministic at
deployment time, which makes "compute as reliable anchor" a load-bearing piece
of the narrative rather than a convenient implementation choice.

### Training

Two-stage: an SFT warmstart on filtered synthetic and bench-derived
trajectories, then GRPO with the composite reward and group-relative advantage
normalisation. The implementation uses TRL + unsloth on a Qwen2.5-7B-Instruct
base, bnb-nf4 4-bit quantised, with LoRA r=32 over q/k/v/o + gate/up/down.
vLLM sleep-mode rollouts share the GPU with the training step so the entire
loop fits on a single 32 GB card.

---

## Datasets — All Public, No IRB

| Dataset | Role | License |
|---|---|---|
| **MedCalc-Bench** (NeurIPS 2024) | Primary in-domain RL training and evaluation; ~55 calculators, ~10K items | CC-BY |
| **MedQA-USMLE** | SFT warmstart + non-tool baseline; 12K items | MIT |
| **MedMCQA** | SFT warmstart augmentation; 194K items | Apache 2.0 |
| **DDXPlus** | Multi-step differential-diagnosis trajectories; 1.3M synthetic cases | CC-BY |
| **MedAgentBench** | Held-out OOD-format tool evaluation; ~300 tasks | Public |
| **Synthetic EHR + multi-tool trajectories** | Self-produced; calculator-verified; target ~30K high-quality trajectories | This work |

### Distribution splits

To support the compositional-generalisation question, MedCalc calculators are
partitioned by clinical specialty:

- **In-domain (train)**: cardiovascular + endocrine + renal (~30 calculators)
- **OOD-tool**: hepatology + neurology + hematology (~25 calculators) —
  never seen during RL training
- **OOD-format**: MedAgentBench tasks with API schemas the model has not
  encountered during training

### Synthetic generation pipeline

1. Sample a calculator and realistic argument values.
2. Compute the gold output by direct execution.
3. Prompt a generator LLM to write a patient note consistent with those
   arguments.
4. **Calc-validate**: re-extract arguments from the note and re-execute;
   keep only trajectories whose recomputed output matches gold.
5. Secondary filter: an LLM judge scores the clinical plausibility of the
   reasoning.

Because `r_compute` is fully deterministic, the contamination risk from
synthetic data is bounded to the `r_select` and `r_args` axes — a controllable
failure mode rather than a hidden one.

---

## Experimental Design

Five experiments are locked against three research questions.

| | Question | Hypothesis |
|---|---|---|
| **RQ1** | Step-level vs. outcome-level verifiable reward in multi-step medical reasoning? | Step-level achieves ≥1.5× sample efficiency |
| **RQ2** | Compositional generalisation to unseen tools and specialties? | Step-level generalisation gap < outcome-level gap |
| **RQ3 ★** | How does verifier degradation affect policy quality and reward hacking? | Step-level remains superior at 30 % verifier noise; both collapse beyond 50 % |

### E1 — Reward Granularity Ablation (RQ1)

Five configurations × 3 seeds, all under V0 oracle verification:

| Run | Reward preset |
|---|---|
| E1-A | SFT-only (no RL) |
| E1-B | outcome |
| E1-C | step_only |
| **E1-D ★** | shaped |
| E1-E | compute_anchored |

**Main figure**: Pareto plot of `accuracy_final` vs.
`avg_tokens_per_trajectory`, with paired-bootstrap significance bars.

### E2 — Compositional Generalisation (RQ2)

Reuses E1's shaped and outcome checkpoints. Evaluates on OOD-tool (unseen
specialties) and OOD-format (MedAgentBench), reporting the generalisation gap
`acc_ID − acc_OOD` as a function of reward granularity.

### E3 — Verifier Degradation (RQ3, paper hero)

The central experiment.

| Run | Reward | Verifier |
|---|---|---|
| E3-1 … E3-3 | shaped  | V0 / V2 / V4 |
| E3-4 … E3-6 | outcome | V0 / V2 / V4 |

3 seeds per cell. **Main figure** (the paper's headline plot): a robustness
curve with verifier noise on the x-axis and `accuracy_final` on the y-axis;
two lines for shaped vs. outcome, shaded by seed std. The hypothesised result
is that shaped RL at 30 % verifier noise still exceeds outcome RL at 0 %.

**Reward-hacking sub-analysis.** Track the rate of trajectories satisfying
`r_select ≥ 0.9 ∧ r_compute < 0.3` (right tool, wrong computation) over
training steps; report two or three case-study trajectories in the paper.

### E4 — Sample Efficiency (RQ1 supplement)

Reuses E1 checkpoints, evaluating every 100 training steps. Learning curves
of `accuracy_final` for shaped vs. outcome (3 seeds shaded). Tests whether
shaped reaches outcome's converged accuracy within the first 30 % of training
steps.

### E5 — Baseline Comparison

Main results table aggregating relevant prior art under a single protocol:

| Method | Type |
|---|---|
| Qwen2.5-7B prompting | Same-base prompting baseline |
| GPT-4o-mini prompting | Closed-source upper-bound reference |
| MedAgents (NeurIPS 2024) | Multi-agent prompting |
| MDAgents (NeurIPS 2024) | Multi-agent prompting |
| Self-Consistency (k=20) | Same-base sampling baseline |
| **Ours (SFT-only)** | This work |
| **Ours (GRPO-shaped)** | This work |

Reported on MedCalc test, OOD-tool, OOD-format, and tokens-per-question.

### Statistical protocol

Every reported number is averaged over seeds {7, 17, 42}. Significance is
assessed with paired bootstrap (10K resamples).

---

## Internal Documents

- `docs/proposal.md` — full research proposal and motivation chain
- `docs/technical_design.md` — formal mathematical definitions of the reward
  functions, GRPO objective, hyperparameters, and TRL + unsloth implementation
  notes
