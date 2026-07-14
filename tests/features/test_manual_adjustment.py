"""Post-fit manual bin adjustment (feature.group) must be reflected by transform:
users who audit and fix a boundary themselves must get consistent output."""

import numpy as np
import pandas as pd
import pytest

from AutoCarver import BinaryCarver
from AutoCarver.combinations import CramervCombinations
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import CategoricalFeature, Features, NestedFeature, OrdinalFeature
from AutoCarver.features.utils.grouped_list import GroupedList


def _six_modality_data() -> tuple[pd.DataFrame, pd.Series]:
    """6 categorical modalities, 300 rows each, distinct exact target rates 0.1..0.6."""
    labels = ["A", "B", "C", "D", "E", "F"]
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    feature, y = [], []
    for label, rate in zip(labels, rates):
        n_pos = round(rate * 300)
        feature += [label] * 300
        y += [1] * n_pos + [0] * (300 - n_pos)
    return pd.DataFrame({"feature": feature}), pd.Series(y)


def test_manual_group_categorical_string_labels():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    a, b = feature.labels[0], feature.labels[1]
    assert a != b

    n_labels_before = len(set(feature.labels))
    assert n_labels_before >= 3

    out_before = carver.transform(X)
    mask_a = out_before["feature"] == a
    mask_b = out_before["feature"] == b
    mask_other = ~(mask_a | mask_b)

    feature.group([a], b)
    out_after = carver.transform(X)

    # a's rows and b's rows now share a single new label, distinct from all others
    assert out_after.loc[mask_a, "feature"].nunique() == 1
    assert out_after.loc[mask_a, "feature"].iloc[0] == out_after.loc[mask_b, "feature"].iloc[0]
    assert out_after["feature"].nunique() == n_labels_before - 1
    # untouched rows keep their exact former label
    assert (out_after.loc[mask_other, "feature"] == out_before.loc[mask_other, "feature"]).all()
    assert len(out_after) == len(X)


def test_manual_group_categorical_ordinal_encoding():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    n_codes_before = len(set(feature.labels))
    a, b = feature.labels[0], feature.labels[1]
    assert a != b

    feature.group([a], b)
    out = carver.transform(X)

    codes = sorted(out["feature"].unique())
    assert codes == list(range(n_codes_before - 1))


def test_manual_group_quantitative_adjacent_intervals():
    """4 quartiles with sharply distinct, monotone target rates -> carving should keep
    all 4 apart (max_n_mod=4); merging two adjacent ones must leave the others alone."""
    rng = np.random.default_rng(0)
    n_per_quartile = 1000
    feature = np.concatenate(
        [
            rng.uniform(0, 25, n_per_quartile),
            rng.uniform(25, 50, n_per_quartile),
            rng.uniform(50, 75, n_per_quartile),
            rng.uniform(75, 100, n_per_quartile),
        ]
    )
    rates = [0.1, 0.3, 0.5, 0.7]
    y = np.concatenate([(rng.random(n_per_quartile) < rate).astype(int) for rate in rates])
    X = pd.DataFrame({"feature": feature})
    y = pd.Series(y)

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature_obj = carver.features("feature")
    n_labels_before = len(set(feature_obj.labels))
    assert n_labels_before >= 3
    label_a, label_b = feature_obj.labels[0], feature_obj.labels[1]

    out_before = carver.transform(X)
    mask_a = out_before["feature"] == label_a
    mask_b = out_before["feature"] == label_b
    mask_other = ~(mask_a | mask_b)

    feature_obj.group([label_a], label_b)
    out_after = carver.transform(X)

    assert out_after.loc[mask_a, "feature"].nunique() == 1
    assert out_after.loc[mask_a, "feature"].iloc[0] == out_after.loc[mask_b, "feature"].iloc[0]
    assert out_after["feature"].nunique() == n_labels_before - 1
    assert (out_after.loc[mask_other, "feature"] == out_before.loc[mask_other, "feature"]).all()
    assert len(out_after) == len(X)


def test_summary_after_manual_group_categorical():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    a, b = feature.labels[0], feature.labels[1]
    before = pd.DataFrame(feature.statistics)

    feature.group([a], b)

    # regression test for the KeyError bug: summary must not raise
    _ = feature.summary

    stats = feature.statistics
    merged_label = feature.labels[0]
    merged_count = before.loc[a, "count"] + before.loc[b, "count"]
    assert stats.loc[merged_label, "count"] == merged_count
    expected_target_mean = (
        before.loc[a, "target_mean"] * before.loc[a, "count"] + before.loc[b, "target_mean"] * before.loc[b, "count"]
    ) / merged_count
    assert stats.loc[merged_label, "target_mean"] == pytest.approx(expected_target_mean)

    # untouched bins keep their exact former rows
    for label in feature.labels[1:]:
        if label == merged_label:
            continue
        assert stats.loc[label, "count"] == before.loc[label, "count"]


def test_summary_after_manual_group_ordinal_encoding():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    n_codes_before = len(set(feature.labels))
    a, b = feature.labels[0], feature.labels[1]

    feature.group([a], b)

    _ = feature.summary  # must not raise

    stats = feature.statistics
    assert sorted(stats.index) == list(range(n_codes_before - 1))


def test_summary_after_manual_group_quantitative():
    rng = np.random.default_rng(0)
    n_per_quartile = 1000
    feature_values = np.concatenate(
        [
            rng.uniform(0, 25, n_per_quartile),
            rng.uniform(25, 50, n_per_quartile),
            rng.uniform(50, 75, n_per_quartile),
            rng.uniform(75, 100, n_per_quartile),
        ]
    )
    rates = [0.1, 0.3, 0.5, 0.7]
    y = np.concatenate([(rng.random(n_per_quartile) < rate).astype(int) for rate in rates])
    X = pd.DataFrame({"feature": feature_values})
    y = pd.Series(y)

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature_obj = carver.features("feature")
    a, b = feature_obj.labels[0], feature_obj.labels[1]
    before = pd.DataFrame(feature_obj.statistics)

    feature_obj.group([a], b)

    _ = feature_obj.summary  # must not raise

    stats = feature_obj.statistics
    merged_label = feature_obj.labels[0]
    merged_count = before.loc[a, "count"] + before.loc[b, "count"]
    assert stats.loc[merged_label, "count"] == merged_count


def test_move_categorical():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    # bin "A" holds ["B", "A"] (a multi-member bin) with max_n_mod=3 on this fixture
    value = "B"
    assert len(feature.content[feature.values.get_group(value)]) > 1
    to_label = "C, D"
    assert to_label != feature.label_per_value.get(value)

    before = pd.DataFrame(feature.statistics)
    out_before = carver.transform(X)
    mask_v = X["feature"] == value
    # rows in the source or target bin (both have their label change); everything else is untouched
    mask_touched = X["feature"].isin(["A", "B", "C", "D"])
    mask_untouched = ~mask_touched

    feature.move(value, to_label)
    out_after = carver.transform(X)

    # value's rows now carry the target bin's label
    assert out_after.loc[mask_v, "feature"].nunique() == 1
    assert out_after.loc[mask_v, "feature"].iloc[0] == "B, C, D"
    # untouched rows (outside the source/target bins) keep their previous transformed label
    assert (out_after.loc[mask_untouched, "feature"] == out_before.loc[mask_untouched, "feature"]).all()

    _ = feature.summary  # must not raise
    stats = feature.statistics
    assert pd.isna(stats.loc["A", "count"])
    assert pd.isna(stats.loc["B, C, D", "count"])
    assert stats.loc["E, F", "count"] == before.loc["E, F", "count"]


def test_move_categorical_ordinal_encoding():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    n_codes_before = len(set(feature.labels))
    value = "B"
    to_label = 1  # int code of a different bin

    feature.move(value, to_label)
    out = carver.transform(X)

    # move relocates a member without eliminating its (still non-empty) source bin,
    # so the number of distinct bins/codes is unchanged
    codes = sorted(out["feature"].unique())
    assert codes == list(range(n_codes_before))
    _ = feature.summary  # must not raise


def test_move_ordinal_contiguity():
    feature = OrdinalFeature("feature", ["low", "medium", "high", "top"])
    feature.group(["low"], "medium", convert_labels=False)
    feature.update_labels()

    # moving "high" into the [low+medium] bin is allowed: contiguous run low-medium-high
    merged_label = feature.labels[0]
    feature.move("high", merged_label)
    assert feature.content[feature.values.get_group("low")] == ["high", "low", "medium"]

    # rebuild and try moving "top" (skips "high") -> not contiguous
    feature = OrdinalFeature("feature", ["low", "medium", "high", "top"])
    feature.group(["low"], "medium", convert_labels=False)
    feature.update_labels()
    merged_label = feature.labels[0]
    with pytest.raises(ValueError):
        feature.move("top", merged_label)


def test_ungroup_categorical():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    before = pd.DataFrame(feature.statistics)
    n_labels_before = len(set(feature.labels))
    value = "B"  # member of the multi-member bin "A, B"
    out_before = carver.transform(X)
    mask_v = X["feature"] == value

    feature.ungroup(value)
    out_after = carver.transform(X)

    assert len(set(feature.labels)) == n_labels_before + 1
    assert out_after.loc[mask_v, "feature"].nunique() == 1
    assert out_after.loc[mask_v, "feature"].iloc[0] != out_before.loc[mask_v, "feature"].iloc[0]

    _ = feature.summary  # must not raise
    stats = feature.statistics
    assert pd.isna(stats.loc["A", "count"])
    assert pd.isna(stats.loc["B", "count"])
    assert stats.loc["E, F", "count"] == before.loc["E, F", "count"]


def _six_modality_nested_data() -> tuple[pd.DataFrame, pd.Series]:
    """Same 6 modalities/target rates as `_six_modality_data`, nested under 3 coarse parents.

    All fine modalities stay frequent (300 rows each), so nested pre-discretization is a
    no-op relabeling and the feature carves exactly like a plain categorical one."""
    fine = ["A", "B", "C", "D", "E", "F"]
    coarse = {"A": "P1", "B": "P1", "C": "P2", "D": "P2", "E": "P3", "F": "P3"}
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    col_fine, col_coarse, y = [], [], []
    for label, rate in zip(fine, rates):
        n_pos = round(rate * 300)
        col_fine += [label] * 300
        col_coarse += [coarse[label]] * 300
        y += [1] * n_pos + [0] * (300 - n_pos)
    return pd.DataFrame({"feature": col_fine, "feature_parent": col_coarse}), pd.Series(y)


def test_move_nested():
    X, y = _six_modality_nested_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(nested={"feature": ["feature_parent"]}),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    assert isinstance(feature, NestedFeature)

    leaders = list(feature.content.keys())
    source_leader = next(leader for leader in leaders if len(feature.content[leader]) > 1)
    target_leader = next(leader for leader in leaders if leader != source_leader)
    value = feature.content[source_leader][0]
    witness = feature.content[target_leader][0]  # stays in target bin, untouched by the move
    to_label = feature.label_per_value.get(target_leader)

    out_before = carver.transform(X)
    mask_touched = X["feature"].isin(feature.content[source_leader] + feature.content[target_leader])
    mask_untouched = ~mask_touched

    feature.move(value, to_label)
    out_after = carver.transform(X)

    # value's rows now share the target bin's (possibly relabelled) label with its witness member
    assert out_after.loc[X["feature"] == value, "feature"].nunique() == 1
    assert (
        out_after.loc[X["feature"] == value, "feature"].iloc[0]
        == out_after.loc[X["feature"] == witness, "feature"].iloc[0]
    )
    # untouched rows (outside the source/target bins) keep their previous transformed label
    assert (out_after.loc[mask_untouched, "feature"] == out_before.loc[mask_untouched, "feature"]).all()

    _ = feature.summary  # must not raise


def test_ungroup_nested():
    X, y = _six_modality_nested_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(nested={"feature": ["feature_parent"]}),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    n_labels_before = len(set(feature.labels))
    source_leader = next(leader for leader, members in feature.content.items() if len(members) > 1)
    value = feature.content[source_leader][0]
    out_before = carver.transform(X)
    mask_v = X["feature"] == value

    feature.ungroup(value)
    out_after = carver.transform(X)

    assert len(set(feature.labels)) == n_labels_before + 1
    assert out_after.loc[mask_v, "feature"].nunique() == 1
    assert out_after.loc[mask_v, "feature"].iloc[0] != out_before.loc[mask_v, "feature"].iloc[0]

    _ = feature.summary  # must not raise


def test_move_nested_unfitted_raises():
    feature = NestedFeature("fine", ["coarse"])
    with pytest.raises(ValueError):
        feature.move("value", "label")
    with pytest.raises(ValueError):
        feature.ungroup("value")


def _quartile_data() -> tuple[pd.DataFrame, pd.Series]:
    """4 quartiles [0,25) [25,50) [50,75) [75,100) with distinct, monotone target rates."""
    rng = np.random.default_rng(0)
    n_per_quartile = 1000
    feature = np.concatenate(
        [
            rng.uniform(0, 25, n_per_quartile),
            rng.uniform(25, 50, n_per_quartile),
            rng.uniform(50, 75, n_per_quartile),
            rng.uniform(75, 100, n_per_quartile),
        ]
    )
    rates = [0.1, 0.3, 0.5, 0.7]
    y = np.concatenate([(rng.random(n_per_quartile) < rate).astype(int) for rate in rates])
    return pd.DataFrame({"feature": feature}), pd.Series(y)


def _fit_quartile_carver(ordinal_encoding: bool = False) -> tuple[BinaryCarver, pd.DataFrame, pd.Series]:
    X, y = _quartile_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=ordinal_encoding, copy=True),
    )
    carver.fit(X, y)
    return carver, X, y


def test_split_quantitative():
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    before = pd.DataFrame(feature.statistics)
    n_labels_before = len(set(feature.labels))
    label = feature.labels[1]  # "(2.50e+01, 5.00e+01]"

    feature.split(label, 30.0)
    out = carver.transform(X)

    assert len(set(out["feature"].unique())) == n_labels_before + 1

    mask_low = (X["feature"] > 25) & (X["feature"] <= 30.0)
    mask_high = (X["feature"] > 30.0) & (X["feature"] <= 50)
    assert out.loc[mask_low, "feature"].nunique() == 1
    assert out.loc[mask_high, "feature"].nunique() == 1
    assert out.loc[mask_low, "feature"].iloc[0] != out.loc[mask_high, "feature"].iloc[0]

    _ = feature.summary  # must not raise
    stats = feature.statistics
    assert pd.isna(stats.loc["(2.50e+01, 3.00e+01]", "count"])
    assert pd.isna(stats.loc["(3.00e+01, 5.00e+01]", "count"])
    assert stats.loc["(-inf, 2.50e+01]", "count"] == before.loc["(-inf, 2.50e+01]", "count"]
    assert stats.loc["(5.00e+01, 7.50e+01]", "count"] == before.loc["(5.00e+01, 7.50e+01]", "count"]
    assert stats.loc["(7.50e+01, inf)", "count"] == before.loc["(7.50e+01, inf)", "count"]


def test_split_quantitative_invalid_point_raises():
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    label = feature.labels[1]  # "(2.50e+01, 5.00e+01]"
    lower_bound = feature.values[0]  # leader of "(-inf, 2.50e+01]", the bin's lower boundary

    with pytest.raises(ValueError):
        feature.split(label, 60.0)  # above the bin's upper bound
    with pytest.raises(ValueError):
        feature.split(label, lower_bound)  # equal to the bin's lower bound


def test_split_ordinal_encoding():
    carver, X, y = _fit_quartile_carver(ordinal_encoding=True)
    feature = carver.features("feature")
    n_codes_before = len(set(feature.labels))
    label = feature.labels[1]

    feature.split(label, 30.0)
    out = carver.transform(X)

    codes = sorted(out["feature"].unique())
    assert codes == list(range(n_codes_before + 1))
    _ = feature.summary  # must not raise


def test_set_boundary_shrink_and_grow():
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    before = pd.DataFrame(feature.statistics)
    label = feature.labels[1]  # "(2.50e+01, 5.00e+01]"

    feature.set_boundary(label, 40.0)  # shrink from ~50
    out = carver.transform(X)
    mask_45 = (X["feature"] > 40.0) & (X["feature"] <= 50.0)
    mask_35 = (X["feature"] > 25.0) & (X["feature"] <= 40.0)
    assert out.loc[mask_45, "feature"].iloc[0] == "(4.00e+01, 7.50e+01]"
    assert out.loc[mask_35, "feature"].iloc[0] == "(2.50e+01, 4.00e+01]"
    assert len(out) == len(X)

    _ = feature.summary  # must not raise
    stats = feature.statistics
    assert pd.isna(stats.loc["(2.50e+01, 4.00e+01]", "count"])
    assert pd.isna(stats.loc["(4.00e+01, 7.50e+01]", "count"])
    assert stats.loc["(-inf, 2.50e+01]", "count"] == before.loc["(-inf, 2.50e+01]", "count"]
    assert stats.loc["(7.50e+01, inf)", "count"] == before.loc["(7.50e+01, inf)", "count"]

    carver2, X2, y2 = _fit_quartile_carver()
    feature2 = carver2.features("feature")
    before2 = pd.DataFrame(feature2.statistics)
    label2 = feature2.labels[1]

    feature2.set_boundary(label2, 60.0)  # grow into the next bin
    out2 = carver2.transform(X2)
    mask_55 = (X2["feature"] > 50.0) & (X2["feature"] <= 60.0)
    mask_65 = (X2["feature"] > 60.0) & (X2["feature"] <= 75.0)
    assert out2.loc[mask_55, "feature"].iloc[0] == "(2.50e+01, 6.00e+01]"
    assert out2.loc[mask_65, "feature"].iloc[0] == "(6.00e+01, 7.50e+01]"

    _ = feature2.summary  # must not raise
    stats2 = feature2.statistics
    assert pd.isna(stats2.loc["(2.50e+01, 6.00e+01]", "count"])
    assert pd.isna(stats2.loc["(6.00e+01, 7.50e+01]", "count"])
    assert stats2.loc["(-inf, 2.50e+01]", "count"] == before2.loc["(-inf, 2.50e+01]", "count"]
    assert stats2.loc["(7.50e+01, inf)", "count"] == before2.loc["(7.50e+01, inf)", "count"]

    with pytest.raises(ValueError):
        feature.set_boundary(feature.labels[-1], 90.0)  # last bin has no upper boundary


def test_split_then_json_roundtrip(tmp_path):
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    feature.split(feature.labels[1], 30.0)
    out_before = carver.transform(X)

    path = tmp_path / "carver.json"
    carver.save(path)
    loaded = BinaryCarver.load(path)
    out_after = loaded.transform(X)

    pd.testing.assert_frame_equal(out_before, out_after)


def test_split_at_interior_member_of_merged_bin():
    """Splitting exactly at a value absorbed by a prior group() must not raise
    (regression: duplicate value inside one GroupedList group)."""
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    label0, label1 = feature.labels[0], feature.labels[1]

    # merging two adjacent bins produces a bin whose content holds a former
    # quantile boundary as an interior (non-leader) member
    feature.group([label0], label1)
    merged_label = feature.labels[0]
    merged_leader = feature._leader_of_label(merged_label)
    interior = min(v for v in feature.values.content[merged_leader] if v != merged_leader)

    feature.split(merged_label, float(interior))  # must not raise

    out = carver.transform(X)
    mask_low = X["feature"] <= float(interior)
    mask_high = (X["feature"] > float(interior)) & (X["feature"] <= merged_leader)
    assert out.loc[mask_low, "feature"].nunique() == 1
    assert out.loc[mask_high, "feature"].nunique() == 1
    assert out.loc[mask_low, "feature"].iloc[0] != out.loc[mask_high, "feature"].iloc[0]

    _ = feature.summary  # must not raise


def _collapse_bin_to_singleton(feature, position: int):
    """Shrinks the bin at ``position`` down to just its leader (content = {leader: [leader]}),
    simulating a carved bin that merged no quantile candidates at fit time. Statistics are
    untouched: label formatting for quantitative features depends on the leader (boundary)
    only, not the member list, so before/after statistics rows stay valid. Returns the leader."""
    leader = list(feature.values)[position]
    new_content = {k: (v if k != leader else [leader]) for k, v in feature.values.content.items()}
    feature.update(GroupedList(new_content), replace=True)
    return leader


def test_split_single_member_bin():
    """Splitting a bin whose content is only its leader ({q: [q]}) must give both
    new bins a NaN stats row instead of a summary KeyError."""
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    before = pd.DataFrame(feature.statistics)
    leader = _collapse_bin_to_singleton(feature, 1)
    assert feature.content[leader] == [leader]

    label = feature.labels[1]  # "(2.50e+01, 5.00e+01]"
    mid = 37.5  # strictly inside the "(25, 50]" quartile bin
    feature.split(label, mid)

    _ = feature.summary  # must not raise (regression: KeyError on synthetic-value-only bin)
    stats = feature.statistics
    assert pd.isna(stats.loc["(2.50e+01, 3.75e+01]", "count"])
    assert pd.isna(stats.loc["(3.75e+01, 5.00e+01]", "count"])
    assert stats.loc["(-inf, 2.50e+01]", "count"] == before.loc["(-inf, 2.50e+01]", "count"]
    assert stats.loc["(5.00e+01, 7.50e+01]", "count"] == before.loc["(5.00e+01, 7.50e+01]", "count"]
    assert stats.loc["(7.50e+01, inf)", "count"] == before.loc["(7.50e+01, inf)", "count"]

    out = carver.transform(X)
    mask_low = (X["feature"] > 25.0) & (X["feature"] <= mid)
    mask_high = (X["feature"] > mid) & (X["feature"] <= 50.0)
    assert out.loc[mask_low, "feature"].nunique() == 1
    assert out.loc[mask_high, "feature"].nunique() == 1
    assert out.loc[mask_low, "feature"].iloc[0] != out.loc[mask_high, "feature"].iloc[0]


def test_set_boundary_single_member_bins():
    carver, X, y = _fit_quartile_carver()
    feature = carver.features("feature")
    before = pd.DataFrame(feature.statistics)
    _collapse_bin_to_singleton(feature, 1)
    label = feature.labels[1]

    feature.set_boundary(label, 40.0)  # shrink

    _ = feature.summary  # must not raise
    stats = feature.statistics
    assert pd.isna(stats.loc["(2.50e+01, 4.00e+01]", "count"])
    assert pd.isna(stats.loc["(4.00e+01, 7.50e+01]", "count"])
    assert stats.loc["(-inf, 2.50e+01]", "count"] == before.loc["(-inf, 2.50e+01]", "count"]
    assert stats.loc["(7.50e+01, inf)", "count"] == before.loc["(7.50e+01, inf)", "count"]

    carver2, X2, y2 = _fit_quartile_carver()
    feature2 = carver2.features("feature")
    before2 = pd.DataFrame(feature2.statistics)
    _collapse_bin_to_singleton(feature2, 1)
    label2 = feature2.labels[1]

    feature2.set_boundary(label2, 60.0)  # grow

    _ = feature2.summary  # must not raise
    stats2 = feature2.statistics
    assert pd.isna(stats2.loc["(2.50e+01, 6.00e+01]", "count"])
    assert pd.isna(stats2.loc["(6.00e+01, 7.50e+01]", "count"])
    assert stats2.loc["(-inf, 2.50e+01]", "count"] == before2.loc["(-inf, 2.50e+01]", "count"]
    assert stats2.loc["(7.50e+01, inf)", "count"] == before2.loc["(7.50e+01, inf)", "count"]


def test_move_colliding_labels_raises():
    """Two 2-member bins whose joined+truncated display labels collide (both render
    "TEST...") must fail loudly on manual editing rather than silently resolving to
    the wrong bin."""
    feature = CategoricalFeature("feature", max_n_chars=4)
    a, b, c, d = "TESTA", "TESTB", "TESTC", "TESTD"
    feature.update(GroupedList({a: [a, b], c: [c, d]}))
    feature.raw_order = [a, b, c, d]
    feature.update_labels()

    assert len(feature.labels) < len(feature.values)  # labels collapsed, values did not

    with pytest.raises(ValueError):
        feature.move(b, feature.labels[0])
