# Reading the GAE dashboard

This guide explains the dashboard produced from `checkpoints/ppo-3/gae.json` in simple terms. The example below shows **PPO update step 1**, **problem 0**, using a counterfactual **gamma of 0.99** and **GAE lambda of 0.90**.

> Important: the critic values in this file are **not vocabulary logits**. A vocabulary-logit vector scores every possible next token. The critic produces only **one scalar value per generated-token position**: its estimate of how much future reward is still expected after the current prefix.

## The example dashboard

[Open the full-size chart](runs/gae_visualizations/step_001_problem_0_counterfactual.png)

![Six-panel token-level GAE dashboard for PPO step 1, problem 0](runs/gae_visualizations/step_001_problem_0_counterfactual.png)

The corresponding per-token numbers are in [the value-list CSV](runs/gae_visualizations/step_001_problem_0_counterfactual.csv).

## The basic idea

The model generates a completion one token at a time. At each token position, PPO asks:

1. **What did the critic expect?** This is the value $V(s_t)$.
2. **Was the result better or worse than expected?** This starts with the TD residual $\delta_t$.
3. **How much credit or blame should this token receive?** This is the advantage $A_t$.
4. **What value should the critic learn instead?** This is the return target $R_t = V(s_t) + A_t$.

A useful mental model is:

- **Value:** the critic's prediction.
- **Reward:** what actually arrived from the environment/reward function.
- **TD residual:** the critic's one-step prediction error or “surprise.”
- **Advantage:** surprise propagated backward through nearby earlier tokens.
- **Return target:** the corrected prediction used to train the critic.

## Terms used in the graph

### Token and token position

A token is a small piece of text, not always a complete word. Position 0 is the first generated token. Moving right follows the generated answer in time. Only a few token labels are printed so the x-axis remains readable.

### Reward $r_t$

The run usually gives most of its reward at the end of a completion. In this example, the total reward is **1.04**:

- exact-answer reward: **1.00**
- `<think>` reward: **0.00**
- length reward: **0.04**

Consequently, the token-reward line is nearly zero until the end. GAE is what carries information from that final outcome backward to earlier tokens.

### Critic value $V(s_t)$

$V(s_t)$ is the critic's estimate of expected future reward after seeing the prompt and generated tokens through position $t$.

- A larger positive value means the critic is optimistic.
- A negative value means the critic is pessimistic.
- It is an estimate, so it can be noisy or badly calibrated.
- A value such as 4 does **not** mean a probability of 4 and does not identify a token.

### Gamma $\gamma$

Gamma discounts outcomes that are farther in the future.

- $\gamma=1$: no time discount.
- Smaller $\gamma$: distant outcomes matter less.
- Larger $\gamma$: reward can influence more distant earlier tokens.

The stored run used $\gamma=1.0$. This displayed chart intentionally recomputes a counterfactual trace with $\gamma=0.99$.

### GAE lambda $\lambda$

Lambda controls how far TD prediction errors are propagated backward.

- $\lambda=0$: advantage uses only the immediate one-step TD residual. It is more local and usually lower variance, but can be more biased.
- $\lambda$ near 1: advantage includes a longer trail of future residuals. It propagates delayed reward farther, but can be noisier.

The stored run used $\lambda=0.95$. This chart recomputes a counterfactual trace with $\lambda=0.90$.

### TD residual $\delta_t$

The one-step critic error is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

Read it as:

> immediate reward + discounted next prediction − current prediction.

- Positive $\delta_t$: the next step looks better than the critic expected.
- Negative $\delta_t$: the next step looks worse than expected.
- Near zero: the two neighboring predictions and reward are consistent.

For the final generated token, there is no next generated state in this trace, so the script uses $V(s_{t+1})=0$.

### Advantage $A_t$

GAE recursively combines the current residual with future residuals:

$$
A_t = \delta_t + \gamma\lambda A_{t+1}.
$$

Equivalently, it is a weighted sum:

$$
A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots.
$$

- Positive advantage: this sampled token/action was followed by a better-than-expected outcome, so PPO tends to increase its probability.
- Negative advantage: it was followed by a worse-than-expected outcome, so PPO tends to decrease its probability.
- Advantage is relative to the critic's expectation. A positive reward does not require every token to have a positive advantage.

### Normalized advantage

PPO normalizes advantages across all valid tokens in the rollout batch. Roughly,

$$
A_t^{\mathrm{norm}} = \frac{A_t-\text{batch mean}}{\text{batch standard deviation}+10^{-8}}.
$$

This changes the scale and center, not the token ordering. A normalized value of $+1$ means approximately one batch standard deviation above the batch mean. This normalized value is what determines the policy update strength before PPO clipping and other loss details.

### Return target

The critic training target is

$$
R_t = A_t + V(s_t).
$$

The old critic predicted $V(s_t)$; GAE says that $R_t$ would be a better target. Critic training tries to move the value prediction toward this target.

## How to read each chart

### 1. Top left — critic values for the complete rollout batch

Each **row** is one generated completion in PPO step 1. Each **column** is a generated-token position.

- Red: positive critic value.
- Blue: negative critic value.
- Pale/white near the center: value near zero.
- Blank white area on the right: the completion already ended; it is padding, not a value of zero.
- Gold horizontal line: the selected example, problem 0.

Use this chart to find batch-wide patterns. For example, a whole row with extreme colors may be an unusual completion, while vertical bands may show that values systematically change around similar generation positions.

**What this chart says:** most critic estimates are positive, but there are isolated negative estimates. Completion lengths vary substantially. The selected row is the first row.

### 2. Top right — old value, return target, and final value

For the selected completion:

- Blue, **Old critic**: $V(s_t)$ before the critic updates.
- Orange, **GAE return target**: $A_t+V(s_t)$, the value the critic is trained toward.
- Green, **Final critic**: value after this step's critic updates.

How to interpret the gaps:

- Blue above orange: the critic was too optimistic relative to the GAE target.
- Blue below orange: the critic was too pessimistic.
- Green moving from blue toward orange: the critic learned in the intended direction.
- Green still near blue: the update was small or did not fit that target well yet.

**What this chart says:** the old critic is very jagged, while the target is much smoother and generally lower. The final green line remains close to the blue line after the limited critic updates, so one step did not make the critic match the target. That is not automatically a bug; critic learning normally happens over many updates.

### 3. Middle left — TD residual decomposition

This chart draws all three pieces of

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

- Blue: immediate token reward $r_t$.
- Orange: discounted next value $\gamma V(s_{t+1})$.
- Green: negative current value $-V(s_t)$.
- Black: their sum, $\delta_t$.

The orange and green lines often largely cancel. The black spikes are positions where adjacent critic predictions are inconsistent or where reward arrives.

**What this chart says:** most immediate token rewards are zero. Much of the black residual therefore comes from large jumps between neighboring critic estimates. Large alternating black spikes indicate a noisy value function across adjacent token prefixes.

### 4. Middle right — raw and normalized advantages

- Blue, **Stored raw advantage**: GAE saved during training with the stored settings $\gamma=1.0$, $\lambda=0.95$.
- Orange, **Counterfactual raw advantage**: recomputed with this chart's $\gamma=0.99$, $\lambda=0.90$.
- Green, **Batch-normalized advantage**: counterfactual advantage normalized over the entire selected rollout batch.

Use the horizontal zero line first:

- Above zero: PPO receives an increase-probability signal.
- Below zero: PPO receives a decrease-probability signal.
- Farther from zero: a stronger unclipped signal.

Do not compare the green line's numeric height directly with blue/orange because green uses the right-hand y-axis and a different scale.

**What this chart says:** reducing gamma and lambda changes early-token credit more than late-token credit because early positions can accumulate many more future residuals. The blue and orange curves are nevertheless broadly similar, so the counterfactual settings preserve much of the sign pattern for this completion.

### 5. Bottom left — how quickly future information fades

A residual $k$ tokens in the future receives weight

$$
(\gamma\lambda)^k.
$$

Here, $\gamma\lambda=0.99\times0.90=0.891$.

- Current residual ($k=0$): weight 1.0.
- About 6 tokens away: weight is roughly one half.
- 10 tokens away: weight is roughly 0.32.
- 20 tokens away: weight is roughly 0.10.
- 40 tokens away: weight is roughly 0.01.

**What this chart says:** with the selected counterfactual settings, nearby future errors matter strongly, but influence fades rapidly over a few dozen tokens. Increasing gamma or lambda would flatten this curve and carry information farther backward.

### 6. Bottom right — sensitivity to gamma and lambda

This panel recomputes the selected completion many times.

- x-axis: candidate lambda.
- y-axis: mean absolute raw advantage, $\operatorname{mean}(|A_t|)$.
- each colored curve: a different gamma.
- thick green curve: selected gamma $0.99$.
- dashed vertical line: selected lambda $0.90$.

This is a **sensitivity diagnostic**, not a score of model quality. Lower is not necessarily better, and the minimum is not automatically the best hyperparameter. Very large magnitudes can make policy updates unstable; very small magnitudes can make the learning signal weak after considering normalization and clipping.

**What this chart says:** for this one completion, advantage magnitude falls as lambda rises through much of the range, then rises sharply close to 1. The sharp rise means long-range residual accumulation is important here. The behavior of one completion is not enough to select hyperparameters; inspect multiple steps and examples and compare training/evaluation stability.

## A practical reading order

When inspecting a new dashboard:

1. Read the title to confirm the **step, problem, gamma, lambda, and reward**.
2. Use the top-left heatmap to see whether the selected example is typical of the batch.
3. In the top-right chart, compare old value with the orange target.
4. Use the middle-left chart to locate where the critic becomes surprised.
5. Use the middle-right chart to see which tokens receive positive or negative policy credit.
6. Use the bottom-left chart to understand the credit-assignment horizon.
7. Treat the bottom-right chart only as a sensitivity warning, not as an automatic hyperparameter chooser.

## Warning signs worth investigating

A single occurrence is not proof of a problem, but repeated patterns across steps can matter:

- critic values become extremely large or drift upward without matching possible returns;
- the final critic repeatedly moves away from the return target;
- neighboring token values alternate wildly throughout every completion;
- almost all normalized advantages have the same sign due to a masking or normalization mistake;
- advantage magnitudes explode as lambda approaches the configured value;
- recomputed values in the CSV disagree materially with stored deltas or advantages when using the stored gamma and lambda.

## Reproduce this exact chart

From the repository root:

```bash
uv run --no-sync python visualizations/plot_gae.py \
  --step 1 \
  --problem-index 0 \
  --gamma 0.99 \
  --gae-lambda 0.90 \
  --tokenizer checkpoints/ppo \
  --output runs/gae_visualizations/step_001_problem_0_counterfactual.png
```

To inspect what is available:

```bash
uv run --no-sync python visualizations/plot_gae.py --list
```

To graph another stored step without counterfactual hyperparameters, omit `--gamma` and `--gae-lambda`:

```bash
uv run --no-sync python visualizations/plot_gae.py --step 1 --example 0 --tokenizer checkpoints/ppo
```

The script also writes a CSV next to each PNG. Use it when an interesting spike needs exact token-level values rather than a visual estimate.
