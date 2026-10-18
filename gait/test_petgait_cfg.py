"""Tests for the pet gait (CPU only).

    python -m pytest gait/test_petgait_cfg.py -v
    python -m gait.test_petgait_cfg              # without pytest installed

The first groups check the reward helpers against numbers worked out by hand; the rest lock the config wiring.
"""

import math

import torch

try:
    import pytest
except ImportError:                     # run without pytest: python -m gait.test_petgait_cfg
    class _Approx:
        def __init__(self, expected, abs=1e-6):
            self.e, self.tol = expected, abs

        def __eq__(self, got):
            if isinstance(self.e, (list, tuple)):
                return len(got) == len(self.e) and all(math.isclose(g, e, rel_tol=1e-6, abs_tol=self.tol) for g, e in zip(got, self.e))
            return math.isclose(got, self.e, rel_tol=1e-6, abs_tol=self.tol)

    class pytest:  # noqa: N801
        approx = _Approx

from gait import mdp_petgait as P
from gait.env_cfg import make_microduck_petgait_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


# ── sigma_of ─────────────────────────────────────────────────────────────────
def test_sigma_k1_matches_cp_tolerance():
    # with k = 1, sigma is exactly the pass criterion's tolerance 0.02 + 10% x command
    s = P.sigma_of(torch.tensor([0.0, 0.05, 0.4]), 1.0)
    assert torch.allclose(s, torch.tensor([0.02, 0.025, 0.06]), atol=1e-6)


def test_sigma_never_zero_at_zero_command():
    # the 0.02 floor: no division by zero at a zero command
    assert P.sigma_of(torch.tensor([0.0]), 1.0).item() > 0


def test_sigma_scales_with_k():
    s1 = P.sigma_of(torch.tensor([0.2]), 1.0).item()
    s8 = P.sigma_of(torch.tensor([0.2]), 8.0).item()
    assert s8 == pytest.approx(8 * s1)


def test_standing_at_005_scores_0018():
    # command 0.05, standing still, k = 1 -> 0.018: standing no longer pays off at slow commands
    s = P.sigma_of(torch.tensor([0.05]), 1.0)
    r = torch.exp(-(0.05 - 0.0) ** 2 / s ** 2).item()
    assert r == pytest.approx(0.018, abs=0.001)


# ── yaw_sigma_of (v5) ────────────────────────────────────────────────────────
def test_yaw_sigma_straight_is_k_times_base():
    # straight (turn command 0), k = 1 -> the base 0.05 rad/s (~3 deg/s)
    assert P.yaw_sigma_of(torch.tensor([0.0]), 1.0).item() == pytest.approx(0.05)
    assert P.yaw_sigma_of(torch.tensor([0.0]), 1.7).item() == pytest.approx(0.085)


def test_yaw_sigma_left_and_right_turn_are_the_same():
    # right (negative) and left (positive) turns of the same size -> same tolerance, wider than straight
    s = P.yaw_sigma_of(torch.tensor([-1.0, 1.0, 0.0]), 1.0)
    assert s[0].item() == pytest.approx(s[1].item())
    assert s[0].item() == pytest.approx(0.15)
    assert s[0].item() > s[2].item()


def test_yaw_sigma_always_positive():
    assert (P.yaw_sigma_of(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]), 1.0) > 0).all()


def test_drift_of_5p5_deg_per_s_scores_about_030():
    # why this base: at k = 1.7, walking straight but drifting 5.5 deg/s (0.096 rad/s) -> 0.30; drifting 1 deg/s -> 0.96
    s = P.yaw_sigma_of(torch.tensor([0.0]), 1.7)
    assert torch.exp(-math.radians(5.5) ** 2 / s ** 2).item() == pytest.approx(0.28, abs=0.03)
    assert torch.exp(-math.radians(1.0) ** 2 / s ** 2).item() == pytest.approx(0.96, abs=0.01)


# ── next_k ───────────────────────────────────────────────────────────────────
def test_k_shrinks_when_good_enough():
    assert P.next_k(8.0, 0.85) == pytest.approx(6.4)       # >= 80% -> shrink to 80%


def test_k_stays_when_not_good_enough():
    assert P.next_k(8.0, 0.79) == pytest.approx(8.0)       # below 80% -> unchanged


def test_k_exactly_at_threshold_shrinks():
    assert P.next_k(8.0, 0.8) == pytest.approx(6.4)        # ">=": exactly 80% counts


def test_k_never_below_min():
    assert P.next_k(1.1, 0.95) == pytest.approx(1.0)       # 1.1 x 0.8 = 0.88 -> held at 1
    assert P.next_k(1.0, 0.99) == pytest.approx(1.0)


def test_k_reaches_one_in_ten_shrinks():
    k = 8.0
    for _ in range(10):
        k = P.next_k(k, 0.9)
    assert k == pytest.approx(1.0)


# ── still_gate ───────────────────────────────────────────────────────────────
def _gate(cmd, speed):
    return P.still_gate(torch.tensor([cmd]), torch.tensor([speed])).item()


def test_gate_on_when_stepping_in_place():
    assert _gate(0.0, 0.004) is True            # command 0, stepping in place (average 0.004) -> penalize


def test_gate_off_while_braking():
    assert _gate(0.0, 0.3) is False             # command 0 but still braking -> no penalty (unavoidable steps)


def test_gate_off_when_commanded_to_move():
    assert _gate(0.05, 0.004) is False          # told to walk (not started yet) -> no penalty
    assert _gate(0.4, 0.3) is False


def test_gate_catches_slow_creep():
    assert _gate(0.0, 0.02) is True             # stepping while creeping -> 0.05 catches it (0.01 would not)


def test_gate_works_on_many_ducks_at_once():
    g = P.still_gate(torch.tensor([0.0, 0.0, 0.4]), torch.tensor([0.004, 0.3, 0.3]))
    assert g.tolist() == [True, False, False]


# ── v10: sigma floor while turning ───────────────────────────────────────────
def test_turn_floor_lifts_in_place_turn():
    # turning in place (linear 0, turn 0.5): sigma raised from k x 0.02 to k x 0.04
    k = 1.074
    s = P.sigma_of(torch.tensor([0.0, 0.03]), k)
    out = P.turn_sigma_floor(s, torch.tensor([0.5, -0.5]), k, floor=0.04)
    assert out.tolist() == pytest.approx([k * 0.04, k * 0.04])   # very slow + turning counts too (no cliff between 0 and 0.03)


def test_turn_floor_leaves_straight_and_standing_alone():
    # no turn command (standing, straight) -> unchanged (C1, C5 unaffected)
    k = 1.074
    s = P.sigma_of(torch.tensor([0.0, 0.05, 0.4]), k)
    out = P.turn_sigma_floor(s, torch.tensor([0.0, 0.0, 0.005]), k, floor=0.04)
    assert torch.equal(out, s)


def test_turn_floor_leaves_fast_walk_and_turn_alone():
    # walking >= 0.2 while turning: already >= k x 0.04, the max changes nothing
    k = 1.074
    s = P.sigma_of(torch.tensor([0.2, 0.4]), k)
    out = P.turn_sigma_floor(s, torch.tensor([0.5, -1.0]), k, floor=0.04)
    assert out.tolist() == pytest.approx(s.tolist())


def test_turn_floor_param_wired(monkeypatch):
    from gait import env_cfg as C
    monkeypatch.setattr(C, "TURN_FLOOR", None)
    assert C.make_microduck_petgait_env_cfg().rewards["track_linear_velocity"].params["turn_floor"] is None
    monkeypatch.setattr(C, "TURN_FLOOR", 0.04)
    assert C.make_microduck_petgait_env_cfg().rewards["track_linear_velocity"].params["turn_floor"] == 0.04


# ── config wiring ────────────────────────────────────────────────────────────
def test_tracking_reward_swapped_and_positive():
    cfg = make_microduck_petgait_env_cfg()
    term = cfg.rewards["track_linear_velocity"]
    assert term.func is P.PetGaitLinVelTracking
    assert term.weight > 0                                   # a reward, not a penalty
    assert term.params["window_s"] == pytest.approx(0.48)


def test_track_weight_override(monkeypatch):
    # TRACK_WEIGHT = None -> as upstream; a number -> only this weight changes, the k parameters stay
    from gait import env_cfg as C
    up = make_microduck_velocity_env_cfg().rewards["track_linear_velocity"].weight
    monkeypatch.setattr(C, "TRACK_WEIGHT", None)
    assert C.make_microduck_petgait_env_cfg().rewards["track_linear_velocity"].weight == up
    monkeypatch.setattr(C, "TRACK_WEIGHT", 3.5)
    term = C.make_microduck_petgait_env_cfg().rewards["track_linear_velocity"]
    assert term.weight == 3.5
    assert term.params["threshold"] == C.K_THRESHOLD


def test_command_is_petgait_with_fast_resample():
    cfg = make_microduck_petgait_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, P.PetGaitVelocityCommandCfg)
    assert cmd.resampling_time_range == (1.5, 3.0)
    assert cmd.rel_slow_straight_envs == pytest.approx(0.15)
    # upstream settings are kept (turning in place, ranges)
    assert cmd.rel_turn_in_place_envs == pytest.approx(0.15)
    from gait import env_cfg as C
    assert cmd.ranges.lin_vel_x == (-0.4, C.MAX_FORWARD)      # backward unchanged, forward limit adjustable
    assert cmd.fast_straight_range == C.FAST_STRAIGHT_RANGE
    assert C.FAST_STRAIGHT_RANGE[1] == C.MAX_FORWARD


def test_everything_else_same_as_velocity():
    # only three changes: every other reward has the velocity task's name and weight
    pet = make_microduck_petgait_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    assert set(pet.rewards) - {"standing_foot_lift", "track_yaw_avg"} == set(vel.rewards)   # v4 and v5 add one term each
    for name in vel.rewards:
        if name == "track_linear_velocity":
            continue
        assert pet.rewards[name].weight == vel.rewards[name].weight, name
    assert set(pet.observations["actor"].terms) == set(vel.observations["actor"].terms)
    assert "petgait_k" in pet.curriculum


def test_yaw_term_only_added_when_weight_set(monkeypatch):
    # YAW_WEIGHT = 0 -> no term (as v4); set -> added, and upstream's track_angular_velocity is untouched
    from gait import env_cfg as C
    monkeypatch.setattr(C, "YAW_WEIGHT", 0.0)
    assert "track_yaw_avg" not in C.make_microduck_petgait_env_cfg().rewards
    monkeypatch.setattr(C, "YAW_WEIGHT", 1.0)
    cfg = C.make_microduck_petgait_env_cfg()
    assert cfg.rewards["track_yaw_avg"].func is P.PetGaitYawTracking
    assert cfg.rewards["track_yaw_avg"].weight == 1.0
    up = make_microduck_velocity_env_cfg().rewards["track_angular_velocity"]
    assert cfg.rewards["track_angular_velocity"].weight == up.weight
    assert cfg.rewards["track_angular_velocity"].params["std"] == up.params["std"]


def test_registered_under_its_own_id():
    # registered at runtime (upstream unmodified); the id must not collide with upstream tasks
    from mjlab.tasks.registry import list_tasks
    from gait.register import TASK_ID
    assert TASK_ID in list_tasks()


if __name__ == "__main__":
    # minimal runner for environments without pytest: each test gets a fresh monkeypatch
    import inspect
    import sys

    class MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, value in reversed(self._undo):
                setattr(obj, name, value)
    bad = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, f in tests:
        mp = MonkeyPatch()
        try:
            f(mp) if "monkeypatch" in inspect.signature(f).parameters else f()
            print(f"ok   {name}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {name}: {e!r}")
        finally:
            mp.undo()
    print(f"\n{len(tests) - bad}/{len(tests)}")
    sys.exit(bad)
