"""EXPERIMENTAL BC-flow actor (DIAG 2026-11-11): deploy path, catalog rules, staging variants, server selection, and the
training-precision rounding of ``OfflinePolicy.act``. CPU fixtures only."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from ops.policy_serving import config as SC
from ops.policy_serving import staging as ST
from s4d.rwlearn import actions as A
from s4d.rwlearn import inference as I
from s4d.rwlearn.deploy import DeployPolicy
from s4d.rwlearn.deploy.fixtures import make_fixture_policy, session_choices, synthetic_window

CONDITIONS = ("reviwo", "sincro", "e2e_cnn")
POSE = np.array([0.2, -0.1, 0.4, 0.1, 0.2, 0.3, 0.9])


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("bcflow")
    out = {}
    for c in CONDITIONS:
        p = make_fixture_policy(root / c, c)
        ckpt, export = root / c / "policy.pt", root / c / "encoder.pt"
        flow = DeployPolicy(ckpt, export_path=export if export.exists() else None, actor="bc_flow")
        out[c] = (p, flow)
    yield out
    torch.set_num_threads(old)


def _window(policy, sid):
    spec = policy.reset_session(**session_choices(policy, sid))
    w = synthetic_window(spec)
    t = w.observation_time_ns
    return w, w.binding("ref", pose_time_ns=t - 3_000_000, gripper_time_ns=t - 40_000_000)


def test_actor_names_agree_and_unknown_actor_is_refused(fixtures):
    assert SC.ACTORS == I.ACTORS == ("onestep", "bc_flow")
    p, _ = fixtures["reviwo"]
    with pytest.raises(ValueError, match="actor"):
        p.offline.sample({"feat": torch.zeros(1, 3, 1)}, None, "argmax")
    with pytest.raises(ValueError, match="actor"):
        DeployPolicy.__init__(DeployPolicy.__new__(DeployPolicy), "unused.pt", actor="bcflow")


@pytest.mark.parametrize("condition", CONDITIONS)
def test_bc_flow_infer_is_the_euler_flow_of_the_same_observation(fixtures, condition):
    one, flow = fixtures[condition]
    noise = torch.linspace(-1, 1, 7)[None]
    w1, ids1 = _window(one, f"one-{condition}")
    w2, ids2 = _window(flow, f"flow-{condition}")
    a1 = one.infer(w1, POSE, 107.5, ids1, noise=noise.numpy())
    a2 = flow.infer(w2, POSE, 107.5, ids2, noise=noise.numpy())
    obs = w2.observation()
    proprio = torch.as_tensor(A.measured_proprio(POSE, 107.5)).float()[None]
    off = flow.offline
    with torch.inference_mode():
        if condition == "e2e_cnn":
            from s4d.rwlearn.pixel import stack_frames
            o = {"pixels": stack_frames(torch.as_tensor(obs["rgb"])[:, 0].unsqueeze(0)), "proprio": proprio}
        else:
            o = {"feat": off.features(obs).to(torch.float16).float(), "proprio": proprio}
        want = off.agent.compute_flow_actions(o, noise).numpy()[0]
        onestep = off.agent.sample_actions(o, noise).numpy()[0]
    np.testing.assert_array_equal(a2["action_norm"], want)
    np.testing.assert_array_equal(a1["action_norm"], onestep)  # same weights: only the actor differs
    assert not np.array_equal(a1["action_norm"], a2["action_norm"])
    assert a2["metadata"]["actor"] == "bc_flow" and a1["metadata"]["actor"] == "onestep"
    assert np.abs(a2["action_norm"]).max() <= 1.0
    assert a2["timings"]["encoder_ms"] > 0 and a2["timings"]["fql_ms"] > 0
    np.testing.assert_allclose(  # the goal is decoded exactly as for the one-step actor
        a2["delta_phys"], flow.offline.contract.denormalise(a2["action_norm"])[:6])


@pytest.mark.parametrize("condition", ("reviwo", "sincro"))
def test_offline_act_rounds_frozen_features_like_training_and_deploy(fixtures, condition):
    one, _ = fixtures[condition]
    w, _ = _window(one, f"round-{condition}")
    obs, noise = w.observation(), torch.linspace(-1, 1, 7)[None]
    deploy = one.act(obs, POSE, 107.5, noise=noise)
    offline = one.offline.act(obs, POSE, 107.5, noise=noise)
    for k in deploy:
        np.testing.assert_array_equal(np.asarray(offline[k]), np.asarray(deploy[k]))


# ------------------------------------------------------------------------------------------------ catalog rules
GOOD = {"checkpoint": "a/policy.pt", "sha256": "0" * 64, "condition": "reviwo", "task": "t", "status": "deployable"}
DECISION = {"decided_by": "user", "at": "2026-11-11T18:00+09:00", "text": "run the bc_flow variants"}


def _cat(policies, aliases=None):
    return {"schema": SC.CATALOG_SCHEMA, "policies": policies, "aliases": aliases or {}}


def test_catalog_bc_flow_entries_are_separate_flagged_and_explicitly_decided():
    vid, v = ST.bcflow_variant("reviwo-t-s0", GOOD)
    assert vid == "reviwo-t-s0+bcflow" and v["actor"] == "bc_flow" and SC.BCFLOW_FLAG in v["flags"]
    assert v["execution_eligible"] is False and v["usage"] == "dry_run_replay_only"
    ok = SC.validate_catalog(_cat({"reviwo-t-s0": GOOD, vid: v}))
    assert ok["policies"][vid]["actor"] == "bc_flow"
    bad = (
        (_cat({"reviwo-t-s0": {**GOOD, "actor": "bc_flow", "flags": [SC.BCFLOW_FLAG]}}), "id ending"),
        (_cat({"x+bcflow": GOOD}), "id ending"),
        (_cat({vid: {**v, "flags": []}}), "flag"),
        (_cat({vid: {**v, "actor": "argmax"}}), "actor"),
        (_cat({"reviwo-t-s0": GOOD, vid: v}, {"reviwo-t": vid}), "experimental"),
        (_cat({vid: {**v, "usage": "execution", "execution_eligible": True,
                     "execution_decision": {**DECISION, "text": "make trained policies eligible"}}}), "naming bc_flow"),
    )
    for raw, match in bad:
        with pytest.raises(ValueError, match=match):
            SC.validate_catalog(raw)
    vid, granted = ST.bcflow_variant("reviwo-t-s0", GOOD, decision=DECISION)
    assert granted["execution_eligible"] is True and granted["execution_decision"] == DECISION
    SC.validate_catalog(_cat({"reviwo-t-s0": GOOD, vid: granted}))
    with pytest.raises(ValueError, match="names bc_flow"):
        ST.bcflow_variant("reviwo-t-s0", GOOD, decision={**DECISION, "text": "yes"})


def test_bc_flow_variants_follow_their_base():
    restricted = {**GOOD, "condition": "dynamo", "status": "restricted", "reasons": ["e1 LN-spread FAIL"]}
    vid, v = ST.bcflow_variant("dynamo-t-s0", restricted, decision=DECISION)
    assert v["status"] == "restricted" and v["execution_eligible"] is False  # never eligible on a restricted base
    entries = {"reviwo-t-s0": GOOD, "reviwo-t-s0+bcflow": ST.bcflow_variant("reviwo-t-s0", GOOD,
                                                                            decision=DECISION)[1]}
    withdrawn = copy.deepcopy(entries)
    withdrawn["reviwo-t-s0"] = {**GOOD, "status": "quarantined", "reasons": ["withdrawn"]}
    synced = ST.sync_bcflow_variants(withdrawn)
    assert synced["reviwo-t-s0+bcflow"]["status"] == "quarantined"
    assert synced["reviwo-t-s0+bcflow"]["execution_eligible"] is False
    del withdrawn["reviwo-t-s0"]
    assert ST.sync_bcflow_variants(withdrawn)["reviwo-t-s0+bcflow"]["status"] == "quarantined"
    assert ST.sync_bcflow_variants(entries) == entries  # idempotent while the base is unchanged
    moved = {**entries, "reviwo-t-s0": {**GOOD, "sha256": "1" * 64}}
    assert ST.sync_bcflow_variants(moved)["reviwo-t-s0+bcflow"]["sha256"] == "1" * 64
    with pytest.raises(ValueError, match="one-step"):
        ST.bcflow_variant("reviwo-t-s0+bcflow", entries["reviwo-t-s0+bcflow"])
    assert ST.legacy_aliases({**entries}) == {"reviwo-t": "reviwo-t-s0"}


def test_server_serves_both_actors_under_their_own_ids(tmp_path):
    from test_policy_serving import client_for, direct, serve_catalog, stage_fixture, write_catalog

    from ops.policy_serving import protocol as P

    torch.set_num_threads(2)
    pol, e = stage_fixture(tmp_path, "r1", "reviwo")
    vid, v = ST.bcflow_variant("reviwo-x-s0", e)
    cat = write_catalog(tmp_path / "catalog.json", {"reviwo-x-s0": e, vid: v}, {"reviwo-x": "reviwo-x-s0"})
    flow = DeployPolicy(tmp_path / "r1/policy.pt", export_path=tmp_path / "r1/encoder.pt", actor="bc_flow")
    with serve_catalog(tmp_path, cat) as srv:
        results = {}
        for pid, ref in (("reviwo-x-s0", pol), (vid, flow)):
            c, feed = client_for(srv, pid, ref)
            for k in range(3):
                feed.add(k)
            results[pid] = (feed.step(2, noise_seed=5), feed)
            c.close()
        assert srv.policies["reviwo-x-s0"].policy.actor == "onestep" and srv.policies[vid].policy.actor == "bc_flow"
        assert srv.policies[vid].info["actor"] == "bc_flow"
    for pid, ref in (("reviwo-x-s0", pol), (vid, flow)):
        r, feed = results[pid]
        assert r.ok, (r.code, r.detail)
        want = direct(ref, feed, 2, noise=P.validation_noise(5))
        np.testing.assert_array_equal(np.asarray(r.outputs["action_norm"]), np.asarray(want["action_norm"]))
    assert not np.array_equal(np.asarray(results["reviwo-x-s0"][0].outputs["action_norm"]),
                              np.asarray(results[vid][0].outputs["action_norm"]))
    assert json.loads(json.dumps(results[vid][0].outputs["metadata"]))["actor"] == "bc_flow"


def test_robot_pc_selection_never_picks_a_bc_flow_variant_unless_asked():
    from ops.robot_pc import selection

    spec = {"condition": "reviwo", "views": 1, "trained_camera_orders": [["c1"]]}
    base = {"input_spec": spec, "info": {"task": "t", "seed": 0, "status": "deployable", "actor": "onestep"}}
    flow = {"input_spec": spec, "info": {"task": "t", "seed": 0, "status": "deployable", "actor": "bc_flow"}}
    offered = {"reviwo-t-s0": base, "reviwo-t-s0+bcflow": flow}
    assert selection.select_policy(offered, condition="reviwo", task="t", seed=0) == "reviwo-t-s0"
    assert selection.select_policy(offered, condition="reviwo", task="t", seed=0,
                                   actor="bc_flow") == "reviwo-t-s0+bcflow"
    with pytest.raises(selection.SelectionError, match="actor"):
        selection.select_policy(offered, "reviwo-t-s0+bcflow")
    with pytest.raises(selection.SelectionError, match="actor"):
        selection.select_policy(offered, "reviwo-t-s0", actor="bc_flow")
    old_server = {"reviwo-t-s0": {"input_spec": spec, "info": {"task": "t", "seed": 0}}}  # no actor label: onestep
    assert selection.select_policy(old_server, condition="reviwo", task="t", seed=0) == "reviwo-t-s0"
