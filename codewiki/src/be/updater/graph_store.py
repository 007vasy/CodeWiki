"""Load and snapshot the saved dependency graph.

``DependencyGraphBuilder.build_dependency_graph`` writes the graph to
``<output>/temp/dependency_graphs/<repo>_dependency_graph.json.gz`` and
overwrites it on every run. The updater therefore copies the previous
graph aside *before* the new graph is built, then loads that copy.

The graph is gzip-compressed JSON: it embeds every component's source
code, so large repositories exceed host file-size limits (GitHub: 100MB)
as plain JSON. Plain ``.json`` graphs from older builds are still read.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil

from codewiki.src.be.dependency_analyzer.models.core import Node

logger = logging.getLogger(__name__)

PREV_SUFFIX = ".prev.json.gz"
GRAPH_SUFFIXES = ("_dependency_graph.json.gz", "_dependency_graph.json")


def sanitize_repo_name(repo_path: str) -> str:
    repo_name = os.path.basename(os.path.normpath(repo_path))
    return "".join(c if c.isalnum() else "_" for c in repo_name)


def graph_file_path(dependency_graph_dir: str, repo_path: str) -> str:
    """Path of the graph for ``repo_path`` inside ``dependency_graph_dir``."""
    return os.path.join(
        dependency_graph_dir, f"{sanitize_repo_name(repo_path)}_dependency_graph.json.gz"
    )


def prev_graph_path(dependency_graph_dir: str, repo_path: str) -> str:
    return graph_file_path(dependency_graph_dir, repo_path)[: -len(".json.gz")] + PREV_SUFFIX


def list_graph_files(dependency_graph_dir: str) -> list[str]:
    """All current graph files in the dir (``.prev`` files excluded).

    gz is the current format; plain ``.json`` files from older builds are
    still listed so they remain findable and prunable.
    """
    if not os.path.isdir(dependency_graph_dir):
        return []
    return sorted(
        os.path.join(dependency_graph_dir, f)
        for f in os.listdir(dependency_graph_dir)
        if f.endswith(GRAPH_SUFFIXES) and ".prev." not in f
    )


def find_any_graph_file(dependency_graph_dir: str) -> str | None:
    """Return the previous build's graph when it is not under the current repo name.

    The repo may have been analysed from a differently named checkout (a git
    worktree per revision, for example), so the sanitized name is not always
    the same. With exactly one candidate that is the answer. With several
    (each earlier worktree left its own file) the newest by mtime is taken and
    the choice is logged; ``prune_superseded_graphs`` keeps this case rare.
    """
    candidates = list_graph_files(dependency_graph_dir)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    newest = max(candidates, key=os.path.getmtime)
    logger.warning(
        "Several dependency graphs found in %s; using the newest (%s) as the previous graph. "
        "Candidates: %s",
        dependency_graph_dir,
        os.path.basename(newest),
        ", ".join(os.path.basename(c) for c in candidates),
    )
    return newest


def prune_superseded_graphs(dependency_graph_dir: str, keep: str) -> list[str]:
    """Delete current-graph files other than ``keep`` so the next update finds one graph.

    The previous graph survives as ``*.prev.json``; only stale copies left by
    earlier checkouts under other names are removed. Returns the removed paths.
    """
    removed = []
    keep_abs = os.path.abspath(keep)
    for path in list_graph_files(dependency_graph_dir):
        if os.path.abspath(path) != keep_abs:
            os.remove(path)
            removed.append(path)
    if removed:
        logger.info(
            "Removed superseded dependency graph(s): %s",
            ", ".join(os.path.basename(r) for r in removed),
        )
    return removed


def snapshot_old_graph(dependency_graph_dir: str, repo_path: str) -> str | None:
    """Copy the previous build's graph to ``*.prev.json.gz``; return that path or None.

    Round-trips through load/save rather than a byte copy so the snapshot
    is valid gzip even when the source is a legacy plain-JSON graph.
    """
    src = graph_file_path(dependency_graph_dir, repo_path)
    if not os.path.exists(src):
        src = find_any_graph_file(dependency_graph_dir)
    if src is None or not os.path.exists(src):
        return None
    dst = prev_graph_path(dependency_graph_dir, repo_path)
    save_graph(load_graph(src), dst)
    logger.info("Saved previous dependency graph to %s", dst)
    return dst


def load_graph(path: str) -> dict[str, Node]:
    """Read a saved graph (gzip or plain JSON) back into ``Node`` objects."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    graph: dict[str, Node] = {}
    for cid, data in raw.items():
        if not isinstance(data, dict):
            continue
        data = dict(data)
        deps = data.get("depends_on") or []
        data["depends_on"] = set(deps)
        data.setdefault("id", cid)
        graph[cid] = Node(**data)
    return graph


def save_graph(graph: dict[str, Node], path: str) -> None:
    """Write ``graph`` in the same shape ``DependencyParser.save_dependency_graph`` uses."""
    result = {}
    for cid, node in graph.items():
        d = node.model_dump()
        if isinstance(d.get("depends_on"), set):
            d["depends_on"] = sorted(d["depends_on"])
        result[cid] = d
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
