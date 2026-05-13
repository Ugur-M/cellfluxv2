"""Tests for ``PlateGroupedSampler``.

Reproducibility contract:

* ``__iter__`` is pure: re-iterating with the same ``_epoch`` yields the
  same sequence. No hidden state mutation.
* ``set_epoch`` is the only writer for the epoch counter; it validates
  type and non-negativity.
* Each epoch emits every dataset position exactly once.
* Within one epoch, all positions for one plate are emitted contiguously
  (the whole point of the plate-grouped sampler).
* Different epochs (with the same seed) produce different shuffles.
* Different seeds (with the same epoch) produce different shuffles.
"""
from __future__ import annotations

import pytest

from cellfluxv2.data.plate_sampler import PlateGroupedSampler


def _make_plates(num_plates: int = 4, per_plate: int = 5) -> dict:
    out: dict = {}
    pos = 0
    for p in range(num_plates):
        out[("expA", p + 1)] = list(range(pos, pos + per_plate))
        pos += per_plate
    return out


# ---------- __iter__ purity (no hidden mutation) ----------------------------

def test_iter_does_not_mutate_epoch():
    """Calling iter twice with no set_epoch yields the same sequence."""
    s = PlateGroupedSampler(_make_plates(), seed=0)
    s.set_epoch(0)
    seq_a = list(s)
    seq_b = list(s)
    assert seq_a == seq_b


def test_iter_full_exhaustion_does_not_advance_epoch():
    """After fully exhausting __iter__, _epoch must still be the set value."""
    s = PlateGroupedSampler(_make_plates(), seed=0)
    s.set_epoch(3)
    list(s)
    assert s.epoch == 3


def test_iter_partial_consumption_does_not_advance_epoch():
    """A partial iter (snapshot-style) also leaves _epoch alone."""
    s = PlateGroupedSampler(_make_plates(), seed=0)
    s.set_epoch(2)
    it = iter(s)
    next(it)
    next(it)
    del it
    assert s.epoch == 2


# ---------- set_epoch is the only writer ------------------------------------

def test_set_epoch_int_updates():
    s = PlateGroupedSampler(_make_plates(), seed=0)
    s.set_epoch(7)
    assert s.epoch == 7


def test_set_epoch_negative_raises():
    s = PlateGroupedSampler(_make_plates(), seed=0)
    with pytest.raises(ValueError, match=">= 0"):
        s.set_epoch(-1)


def test_set_epoch_bool_raises():
    s = PlateGroupedSampler(_make_plates(), seed=0)
    with pytest.raises(ValueError, match="int"):
        s.set_epoch(True)  # type: ignore[arg-type]


def test_set_epoch_float_raises():
    s = PlateGroupedSampler(_make_plates(), seed=0)
    with pytest.raises(ValueError, match="int"):
        s.set_epoch(1.0)  # type: ignore[arg-type]


# ---------- coverage --------------------------------------------------------

def test_every_position_emitted_exactly_once_per_epoch():
    plates = _make_plates(num_plates=4, per_plate=5)
    expected = sorted(p for positions in plates.values() for p in positions)
    s = PlateGroupedSampler(plates, seed=0)
    s.set_epoch(0)
    got = sorted(list(s))
    assert got == expected


def test_len_matches_total_positions():
    plates = _make_plates(num_plates=3, per_plate=7)
    s = PlateGroupedSampler(plates, seed=0)
    assert len(s) == 3 * 7


# ---------- plate-contiguity invariant --------------------------------------

def test_positions_for_same_plate_are_contiguous_in_one_epoch():
    plates = _make_plates(num_plates=4, per_plate=5)
    pos_to_plate: dict[int, tuple[str, int]] = {}
    for plate_key, positions in plates.items():
        for p in positions:
            pos_to_plate[p] = plate_key

    s = PlateGroupedSampler(plates, seed=0)
    s.set_epoch(0)
    seq = list(s)

    # Walk the sequence and record plate transitions.
    transitions: list[tuple[str, int]] = []
    last_plate: tuple[str, int] | None = None
    for p in seq:
        plate = pos_to_plate[p]
        if plate != last_plate:
            transitions.append(plate)
            last_plate = plate
    # If positions are contiguous per plate, every plate appears exactly
    # once across transitions.
    assert len(transitions) == len(plates), (
        f"plates not contiguous; transitions: {transitions}"
    )
    assert set(transitions) == set(plates.keys())


# ---------- determinism: same epoch + same seed -> same sequence ------------

def test_same_seed_same_epoch_same_sequence_across_instances():
    plates = _make_plates()
    a = PlateGroupedSampler(plates, seed=42)
    b = PlateGroupedSampler(plates, seed=42)
    a.set_epoch(5)
    b.set_epoch(5)
    assert list(a) == list(b)


def test_different_epoch_different_sequence():
    plates = _make_plates()
    s = PlateGroupedSampler(plates, seed=42)
    s.set_epoch(0)
    seq_e0 = list(s)
    s.set_epoch(1)
    seq_e1 = list(s)
    assert seq_e0 != seq_e1


def test_different_seed_different_sequence():
    plates = _make_plates()
    a = PlateGroupedSampler(plates, seed=0)
    b = PlateGroupedSampler(plates, seed=1)
    a.set_epoch(0)
    b.set_epoch(0)
    assert list(a) != list(b)
