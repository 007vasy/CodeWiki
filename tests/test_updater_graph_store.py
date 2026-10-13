"""Previous-graph lookup when the repo was analysed from differently named checkouts.

The graph is stored as gzip JSON (``*_dependency_graph.json.gz``); legacy
plain ``.json`` graphs from older builds remain readable.
"""

from __future__ import annotations

import gzip
import json
import os
import time

from codewiki.src.be.updater.graph_store import (
    find_any_graph_file,
    graph_file_path,
    load_graph,
    prune_superseded_graphs,
    save_graph,
    snapshot_old_graph,
)

_COMPONENT = {
    "id": "c",
    "name": "c",
    "component_type": "class",
    "file_path": "f",
    "relative_path": "f",
}


def _node(cid):
    d = dict(_COMPONENT)
    d["id"] = d["name"] = cid
    return d


def _write_graph(d, name, payload, ext=".json.gz"):
    """Write a graph file the way DependencyParser saves them (gz default,
    plain for legacy)."""
    path = os.path.join(d, f"{name}_dependency_graph{ext}")
    if ext == ".json.gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    return path


def test_single_candidate_is_used_whatever_its_name(tmp_path):
    d = str(tmp_path)
    old = _write_graph(d, "rq3_svelte_anchor", {"a": _node("a")})
    assert find_any_graph_file(d) == old
    prev = snapshot_old_graph(d, "/repos/rq3-svelte-w1")
    assert prev is not None and prev.endswith("rq3_svelte_w1_dependency_graph.prev.json.gz")
    loaded = load_graph(prev)
    assert loaded["a"].id == "a"


def test_legacy_plain_json_graph_is_found_and_snapshotted(tmp_path):
    d = str(tmp_path)
    _write_graph(d, "rq3_svelte_anchor", {"a": _node("a")}, ext=".json")
    prev = snapshot_old_graph(d, "/repos/rq3-svelte-w1")
    # A plain-JSON source still snapshots to valid gzip JSON.
    assert prev is not None and prev.endswith(".prev.json.gz")
    with gzip.open(prev, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["a"]["id"] == "a"


def test_several_candidates_pick_newest_not_none(tmp_path):
    d = str(tmp_path)
    older = _write_graph(d, "rq3_svelte_anchor", {"rev": _node("anchor")})
    time.sleep(0.01)
    newer = _write_graph(d, "rq3_svelte_w1", {"rev": _node("w1")})
    os.utime(older, (1, 1))  # make the order unambiguous
    assert find_any_graph_file(d) == newer
    prev = snapshot_old_graph(d, "/repos/rq3-svelte-w2")
    assert prev is not None
    assert load_graph(prev)["rev"].name == "w1"


def test_prev_snapshots_are_not_candidates(tmp_path):
    d = str(tmp_path)
    _write_graph(d, "rq3_svelte_w1", {"rev": _node("w1")})
    with gzip.open(
        os.path.join(d, "rq3_svelte_w1_dependency_graph.prev.json.gz"), "wt", encoding="utf-8"
    ) as f:
        json.dump({"rev": _node("anchor")}, f)
    assert find_any_graph_file(d).endswith("rq3_svelte_w1_dependency_graph.json.gz")


def test_prune_keeps_current_and_prev_only(tmp_path):
    d = str(tmp_path)
    _write_graph(d, "rq3_svelte_anchor", {})
    _write_graph(d, "rq3_svelte_w1", {})
    _write_graph(d, "rq3_svelte_w2", {}, ext=".json")  # legacy plain current
    current = _write_graph(d, "rq3_svelte_w3", {})
    with gzip.open(
        os.path.join(d, "rq3_svelte_w3_dependency_graph.prev.json.gz"), "wt", encoding="utf-8"
    ) as f:
        json.dump({}, f)
    removed = prune_superseded_graphs(d, keep=graph_file_path(d, "/repos/rq3-svelte-w3"))
    assert sorted(os.path.basename(r) for r in removed) == [
        "rq3_svelte_anchor_dependency_graph.json.gz",
        "rq3_svelte_w1_dependency_graph.json.gz",
        "rq3_svelte_w2_dependency_graph.json",
    ]
    assert sorted(os.listdir(d)) == [
        "rq3_svelte_w3_dependency_graph.json.gz",
        "rq3_svelte_w3_dependency_graph.prev.json.gz",
    ]
    assert os.path.exists(current)


def test_save_and_load_graph_roundtrip_gzip(tmp_path):
    from codewiki.src.be.dependency_analyzer.models.core import Node

    path = os.path.join(str(tmp_path), "g_dependency_graph.json.gz")
    node = Node(id="x", name="x", component_type="class", file_path="f", relative_path="f")
    save_graph({"x": node}, path)
    assert path.endswith(".json.gz")
    loaded = load_graph(path)
    assert loaded["x"].id == "x" and isinstance(loaded["x"].depends_on, set)
