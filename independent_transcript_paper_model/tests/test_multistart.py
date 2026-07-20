import numpy as np

from hierarchical_confidence.fit import ParameterTransform
from hierarchical_confidence.multistart import (
    generate_multistart_parameters,
    make_multistart_schedule,
    parameters_from_record,
)


COHERENCES = np.array([0.06, 0.12, 0.24])


def test_multistart_generation_is_reproducible_and_starts_with_default() -> None:
    first = generate_multistart_parameters(COHERENCES, n_starts=4, seed=17)
    second = generate_multistart_parameters(COHERENCES, n_starts=4, seed=17)

    default = ParameterTransform(COHERENCES).default_parameters()
    assert first[0].rho == default.rho
    np.testing.assert_allclose(first[0].sensory_kappas, default.sensory_kappas)
    assert first[0].motor_kappa == default.motor_kappa
    assert first[0].lapse == default.lapse
    for left, right in zip(first, second):
        assert left.rho == right.rho
        np.testing.assert_allclose(left.sensory_kappas, right.sensory_kappas)
        assert left.motor_kappa == right.motor_kappa
        assert left.lapse == right.lapse


def test_random_starts_are_valid_and_begin_with_monotonic_sensory_precision() -> None:
    starts = generate_multistart_parameters(COHERENCES, n_starts=7, seed=23)

    assert len(starts) == 7
    for parameters in starts:
        parameters.validate(len(COHERENCES))
        assert np.all(np.diff(parameters.sensory_kappas) > 0.0)


def test_schedule_records_round_trip_to_parameters() -> None:
    schedule = make_multistart_schedule([3, 5], COHERENCES, n_starts=3, base_seed=101)

    assert len(schedule) == 6
    assert schedule.groupby("subject_id")["start_id"].nunique().eq(3).all()
    row = schedule.iloc[-1]
    parameters = parameters_from_record(row, COHERENCES, prefix="initial_")
    np.testing.assert_allclose(
        parameters.sensory_kappas,
        [
            row["initial_sensory_kappa_coherence_0.06"],
            row["initial_sensory_kappa_coherence_0.12"],
            row["initial_sensory_kappa_coherence_0.24"],
        ],
    )
