# Tie-Aware MAP Readout

## Why this correction is required

The original observer explanation normalizes each measurement-conditional posterior, rounds it to six decimal places, and treats every maximum as an equally probable percept. A single `argmax` silently selects one grid position when two modes have equal height, which can remove a legitimate branch from predicted response distributions.

This matters directly for the project question because bimodality must emerge from the model rather than from numerical tie breaking.

## Declared rule

For each possible sensory measurement:

1. Normalize the posterior over direction.
2. Round normalized probabilities to six decimal places.
3. Identify all bins equal to the rounded maximum.
4. Assign each tied bin probability `1 / number_of_tied_bins`.
5. Integrate this readout distribution over sensory measurements.
6. Apply circular motor noise and lapse.

The rule is deterministic. Fitting never randomly chooses one tied mode.

## Shared behavior

Likelihood scoring and saved response predictions use the same tie-aware calculation. The PyTorch/CUDA backend follows the same normalization, rounding, and equal-mass rule as the NumPy/SciPy reference.

The notebooks save diagnostics containing:

- total trial-measurement posterior cases;
- number and unweighted rate of tied cases;
- expected tie probability after weighting measurements by `P(m | true direction)`;
- maximum number of tied MAP bins.

## Interpretation

Tie-aware MAP does not force bimodality. It only prevents equal posterior modes from being collapsed by array-index order. Bimodality remains an empirical model prediction that must be checked against human response distributions.

Final scientific results should also assess sensitivity to angular resolution and to five-, six-, and seven-decimal rounding. Six decimals is the declared primary analysis because it follows the original-model explanation.
