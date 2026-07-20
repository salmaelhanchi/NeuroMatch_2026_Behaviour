# Independent Technical Map

## 1. Project question

The original paper asks whether human direction estimates come from multiplying a sensory likelihood by a learned prior or from switching between them.

The project adds a new question:

> Can a hierarchical belief about prior confidence learn across trials and blocks, while the prior mean remains fixed, and explain both unimodal and bimodal response distributions?

This is an extension of the paper, not a restatement of it.

## 2. Source-derived decisions

| Decision | Source status | Consequence |
|---|---|---|
| Direction is circular | Paper-defined | Use a 1-to-360-degree circular grid and wrapped errors |
| Sensory evidence is noisy | Paper-defined | Introduce a latent measurement instead of using true direction as the observer's measurement |
| Prior mean is 225 degrees | Paper-defined and TA-confirmed | Never update the mean |
| Prior confidence varies | TA-directed | Learn concentration `kappa`, where larger values mean a narrower, stronger prior |
| Prior confidence is hidden | Project hypothesis | Maintain a probability distribution `H_t(kappa)` rather than assuming one known concentration |
| Peaked and uniform components are possible | Secondary PDF and paper Discussion | Reserve this as an optional extension if hidden confidence alone is insufficient |
| State carries across blocks | TA-directed | Last state of one block initializes the next block |
| Do not add a session-level hierarchy initially | TA-directed | Keep the first model at trial and block levels |
| Fit subjects individually | TA-directed and paper-defined | Produce 12 independent fits before group summaries |
| Validate before AIC | TA-directed | Plot observed and predicted distributions and perform recovery first |
| Compare against original Switching observer | TA-directed | Switching is a required comparator |

## 3. Experimental map

### Trial input

Each trial contains:

- true motion direction `d_t`;
- motion coherence `c_t`, controlling sensory reliability;
- block condition with a direction distribution centered at 225 degrees;
- participant response `e_t`;
- true-direction feedback after the response.

### Block structure

Within a block, directions are sampled from one of four prior widths. The center remains 225 degrees. The observer must infer confidence from the experienced directions and feedback, not from the experimenter's prior-width label.

### Subject structure

The architecture is shared, but model parameters can differ by subject. Group results summarize subject-level fits; they do not replace them.

## 4. Circular representation

Define signed angular difference as:

```text
wrap(x) = ((x + 180) mod 360) - 180
```

For every trial:

```text
response_angle = atan2(estimate_y, estimate_x) mod 360
stimulus_relative = wrap(d_t - 225)
response_relative = wrap(e_t - 225)
estimation_error = wrap(e_t - d_t)
```

Ordinary linear subtraction is invalid near the 0/360 boundary.

## 5. Original paper models

### 5.1 Sensory measurement

The observer does not directly see `d_t`. It receives a noisy measurement:

```text
m_t ~ VM(d_t, kappa_sensory[c_t])
```

Higher coherence should produce larger sensory concentration.

### 5.2 Basic Bayesian observer

The fixed prior is:

```text
P_prior(theta) = VM(theta; 225, kappa_prior[condition])
```

For a latent measurement `m`:

```text
P(theta | m) proportional to VM(theta; m, kappa_sensory) * P_prior(theta)
```

The paper uses a posterior readout, then motor noise and lapses. The response probability for known true direction integrates over all possible measurements:

```text
P(e | d) = sum_m P(e | m) P(m | d)
```

Expected behavior: one response mode between the likelihood and prior when they conflict.

### 5.3 Switching observer

The paper's Switching observer does not multiply the two sources into one estimate. It chooses a prior-based or sensory-based estimate with probabilities determined by relative concentration:

```text
P(prior branch) = kappa_prior / (kappa_prior + kappa_sensory)
P(sensory branch) = 1 - P(prior branch)
```

Expected behavior: two modes can appear, one near 225 degrees and one near sensory evidence.

### 5.4 Shared observation model

Every model must use the same final response process:

```text
P_reported = (1 - lapse) * circular_convolution(P_percept, motor_noise)
             + lapse * Uniform(360)
```

This ensures comparison tests the inference rule rather than different noise assumptions.

## 6. Target hierarchical confidence model

### 6.1 What changes

The target model replaces four fixed subjective prior concentrations with an evolving belief about concentration:

```text
H_t(kappa) = belief about prior concentration before trial t
```

The prior mean is always 225 degrees.

### 6.2 Hyper-state prediction

To remain adaptive when blocks change, previous evidence must be discounted or diffused. A minimal implementation choice is:

```text
H_t_minus(kappa) proportional to H_t(kappa)^rho
```

where `rho` is a subject-level memory parameter between 0 and 1.

- `rho` near 1 means long memory and slow adaptation.
- smaller `rho` means faster forgetting and adaptation.

This exact discount equation is an implementation choice. The source requirement is only that confidence updates and carries between blocks.

### 6.3 Effective prior over direction

Average the possible prior concentrations under the current hyper-state:

```text
P_effective_t(theta)
  = sum_kappa H_t_minus(kappa) VM(theta; 225, kappa)
```

This is the participant's prior over direction after uncertainty about prior confidence has been marginalized. It is one prior distribution, not a selected branch.

### 6.4 One integrated posterior

For each possible sensory measurement `m`, combine the likelihood with the effective prior:

```text
P(theta | m, H_t)
  proportional to VM(theta; m, kappa_sensory)
                  * P_effective_t(theta)
```

Normalize this into one posterior and apply one readout rule. The response prediction then integrates that readout over all possible sensory measurements.

No peaked-versus-uniform branch is selected in the primary model. Any switching-like response structure must emerge from uncertainty over `kappa`, sensory uncertainty, posterior shape, and the readout.

### 6.5 What the bimodality test means

The project hypothesis is that the evolving hidden confidence may reproduce both unimodal and bimodal response distributions without an explicit selection rule. This must be tested, not assumed.

- If the validated model produces and fits both shapes, hidden confidence provides a possible integrated explanation.
- If it fits ordinary prior attraction but cannot generate the observed two modes, the simplest hidden-confidence hypothesis is insufficient.
- That negative result is scientifically meaningful and should not be repaired by silently adding a switching branch.

### 6.6 Feedback update

After the participant responds, the revealed true direction is evidence about prior concentration:

```text
H_{t+1}(kappa)
  proportional to VM(d_t; 225, kappa) * H_t_minus(kappa)
```

Then normalize over `kappa`.

This update:

- keeps the mean fixed;
- increases support for high concentration after directions near 225 degrees;
- increases support for low concentration after directions far from 225 degrees;
- uses information available to the participant;
- never reads `prior_std`.

The update occurs after predicting trial `t`, so it affects trial `t+1`.

### 6.7 Block carryover

At a block boundary:

```text
first hyper-state of new block = final hyper-state of previous block
```

No instantaneous reset to the new true prior width is permitted. The new block influences confidence only through its revealed directions.

Whether memory carries across separate sessions is not fixed by the sources. The main analysis should state one rule in advance and test the alternative as a sensitivity analysis.

### 6.8 Optional two-regime confidence extension

Only after the primary hidden-confidence model is implemented, validated, and recovered may the concentration belief be extended to represent distinct low- and high-confidence regimes, including mass near `kappa = 0`.

Even in this extension, first marginalize over `kappa` to create one effective prior and then form one posterior. Reading out low- and high-confidence branches separately would create a gating model and must be labeled as a separate switching-like candidate.

## 7. First parameter set

Fit separately for each subject:

| Parameter | Meaning | Constraint |
|---|---|---|
| `kappa_6` | Sensory concentration at 6% coherence | Positive |
| `kappa_12` | Sensory concentration at 12% coherence | Greater than or equal to `kappa_6` |
| `kappa_24` | Sensory concentration at 24% coherence | Greater than or equal to `kappa_12` |
| `rho` | Hyper-state memory/discount | Between 0 and 1 |
| `kappa_motor` | Motor precision | Positive |
| `lapse` | Uniform response probability | Between 0 and 1 |

Keep the initial hyper-state fixed and broad in the first recovery experiment. Fit an initial-state parameter only if recovery demonstrates it is distinguishable from `rho`.

## 8. Required comparators

| Model | Role |
|---|---|
| Sensory-only | Establishes whether prior information is needed |
| Basic Bayesian | Original multiplicative-integration baseline |
| Switching | Original paper comparator |
| Hierarchical hidden-confidence observer | New project model with one integrated posterior |

All candidates receive identical trials and use the same motor and lapse definitions.

## 9. Validation requirements from the TA

Before AIC or interpretation:

1. Plot each subject's observed distributions.
2. Plot model-predicted distributions for the same conditions.
3. Confirm that the hierarchical state converges rather than jumping noisily.
4. Simulate unimodal and bimodal cases.
5. Perform parameter recovery.
6. Fit each of the 12 subjects separately.
7. Only then perform model comparison.

## 10. Claims the design can and cannot support

It can test whether a feedback-driven confidence state improves predictions and captures distribution shapes.

It cannot establish that the brain literally represents the mathematical hyper-state. A successful result supports the computational hypothesis, while the Switching observer may remain a simpler biological implementation.

