"""Tests for the tree-sitter based Go analyzer."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_go")

from codewiki.src.be.dependency_analyzer.analyzers.go import analyze_go_file
from codewiki.src.be.dependency_analyzer.ast_parser import DependencyParser

SAMPLE = """\
package services

import (
	"fmt"

	"github.com/smartcontractkit/chainlink-common/pkg/types"
)

// Server serves things.
type Server struct {
	Base
	id    int
	store *Store
}

type Base struct{}

// Handler handles requests.
type Handler interface {
	Handle(ctx context.Context) error
	fmt.Stringer
}

// Start starts the server.
func (s *Server) Start(ctx context.Context, cfg Config) error {
	helper(ctx)
	s.store.Save()
	local := NewWorker()
	fmt.Sprintf("%v", local)
	return nil
}

func helper(ctx context.Context) {}

func NewWorker() *Worker { return nil }

func (st *Store) Save() error { return nil }

type Worker struct{}
type Store struct{}
type Config struct{}
"""

# A second "package": same go.mod module, different directory, so the
# project-import machinery classifies the import as internal.
OTHER_PKG = """\
package workers

// Worker runs jobs.
type Worker struct{}

func (w *Worker) Run() error { return nil }

func Spawn() *Worker { return nil }
"""


def _go_repo(tmp_path: Path) -> Path:
    (tmp_path / "go.mod").write_text(
        "module github.com/example/mega\n\ngo 1.22\n", encoding="utf-8"
    )
    return tmp_path


def _analyze(tmp_path: Path, sample: str = SAMPLE, name: str = "server.go", sub: str = ""):
    file_path = tmp_path / sub / name if sub else tmp_path / name
    if sub:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    file_path.write_text(sample, encoding="utf-8")
    return analyze_go_file(str(file_path), sample, repo_path=str(tmp_path))


def test_extracts_structs_interfaces_functions_and_methods(tmp_path: Path) -> None:
    _go_repo(tmp_path)
    nodes, _ = _analyze(tmp_path)
    by_name = {node.name: node for node in nodes}

    assert by_name["Server"].component_type == "class"
    assert by_name["Server"].node_type == "struct"
    assert by_name["Server"].base_classes == ["Base"]
    assert by_name["Base"].component_type == "class"

    assert by_name["Handler"].component_type == "interface"
    assert by_name["Handler"].node_type == "interface"
    assert by_name["Handler"].base_classes == ["Stringer"]

    assert by_name["Server.Start"].component_type == "method"
    assert by_name["Server.Start"].class_name == "Server"
    assert by_name["helper"].component_type == "function"
    assert by_name["NewWorker"].component_type == "function"

    assert by_name["Server"].id == "server.go::Server"
    assert by_name["Server.Start"].id == "server.go::Server.Start"
    assert all(node.language == "go" for node in nodes)


def test_extracts_docstring_and_parameters(tmp_path: Path) -> None:
    _go_repo(tmp_path)
    nodes, _ = _analyze(tmp_path)
    by_name = {node.name: node for node in nodes}

    assert by_name["Server"].has_docstring
    assert "serves things" in by_name["Server"].docstring
    assert by_name["Server.Start"].parameters == ["ctx context.Context", "cfg Config"]
    # Qualified names feed the global resolver's package-qualified matching.
    assert by_name["Server.Start"].qualified_name == "services.Server.Start"
    assert by_name["helper"].qualified_name == "services.helper"


def test_extracts_call_relationships(tmp_path: Path) -> None:
    _go_repo(tmp_path)
    _, relationships = _analyze(tmp_path)
    edges = {(rel.caller, rel.callee, rel.is_resolved) for rel in relationships}

    caller = "server.go::Server.Start"
    # Bare call resolves to the same-file function.
    assert (caller, "server.go::helper", True) in edges
    # Chained selector through an in-file struct field resolves to the method.
    assert (caller, "server.go::Store.Save", True) in edges
    # Composite literal ``x := NewWorker()`` stays a call edge only if it is a
    # call; here the receiver is a function call result, so no edge fires.
    # Struct field type edge from the struct itself.
    assert ("server.go::Server", "server.go::Store", True) in edges
    # Embedded struct: extends-style edge, resolved within the same file.
    assert ("server.go::Server", "server.go::Base", True) in edges
    # Parameter type edge from the method signature.
    assert (caller, "server.go::Config", True) in edges

    # External package calls are dropped entirely, not unresolved noise.
    assert not any(rel.callee.startswith("fmt.") for rel in relationships)
    # Builtin noise is excluded.
    assert not any(rel.callee == "Sprintf" for rel in relationships)


def test_project_import_calls_emit_qualified_candidates(tmp_path: Path) -> None:
    repo = _go_repo(tmp_path)
    sample = """\
package api

import (
	workers "github.com/example/mega/pkg/workers"
)

type Client struct{}

func (c *Client) Boot() error {
	workers.Spawn()
	return nil
}
"""
    (repo / "pkg" / "workers").mkdir(parents=True)
    (repo / "pkg" / "workers" / "worker.go").write_text(OTHER_PKG, encoding="utf-8")
    _, relationships = _analyze(tmp_path, sample=sample, name="client.go")

    edges = {(rel.caller, rel.callee, rel.is_resolved) for rel in relationships}
    # Project-internal import: the qualified candidate is emitted unresolved;
    # global resolution ties it to workers.Spawn via qualified_name matching.
    assert ("client.go::Client.Boot", "workers.Spawn", False) in edges


def test_external_import_calls_are_dropped(tmp_path: Path) -> None:
    _go_repo(tmp_path)
    sample = """\
package api

import "github.com/other/repo/thing"

type Client struct{}

func (c *Client) Boot() error {
	thing.DoThing()
	return nil
}
"""
    _, relationships = _analyze(tmp_path, sample=sample, name="client.go")
    assert not any("DoThing" in rel.callee for rel in relationships)


def test_no_go_mod_means_all_imports_external(tmp_path: Path) -> None:
    # No go.mod: nothing can be classified project-internal.
    sample = """\
package api

import "github.com/example/mega/pkg/workers"

type Client struct{}

func (c *Client) Boot() error {
	workers.Spawn()
	return nil
}
"""
    _, relationships = _analyze(tmp_path, sample=sample, name="client.go")
    assert not any("Spawn" in rel.callee for rel in relationships)


def test_empty_file_returns_no_components(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.go"
    file_path.write_text("", encoding="utf-8")
    nodes, relationships = analyze_go_file(str(file_path), "", repo_path=str(tmp_path))
    assert nodes == []
    assert relationships == []


def test_go_extension_reaches_the_go_analyzer(tmp_path: Path) -> None:
    repo = _go_repo(tmp_path)
    sample = """\
package main

type App struct{}

func (a *App) Run() error { return nil }

func New() *App { return nil }
"""
    (repo / "app.go").write_text(sample, encoding="utf-8")
    components = DependencyParser(str(repo)).parse_repository()
    assert "app.go::App" in components
    assert "app.go::App.Run" in components
    assert "app.go::New" in components


def test_dependency_parser_end_to_end_cross_package(tmp_path: Path) -> None:
    repo = _go_repo(tmp_path)
    (repo / "api").mkdir()
    (repo / "api" / "client.go").write_text(
        """\
package api

import "github.com/example/mega/pkg/workers"

type Client struct{ w *workers.Worker }

func (c *Client) Boot() error {
	workers.Spawn()
	return nil
}
""",
        encoding="utf-8",
    )
    (repo / "pkg" / "workers").mkdir(parents=True)
    (repo / "pkg" / "workers" / "worker.go").write_text(OTHER_PKG, encoding="utf-8")

    components = DependencyParser(str(repo)).parse_repository()

    assert "api/client.go::Client" in components
    assert "api/client.go::Client.Boot" in components
    assert "pkg/workers/worker.go::Worker" in components

    # The qualified candidate `workers.Spawn` resolves during global
    # resolution (qualified_name `workers.Spawn`, matched on its last two
    # parts) and becomes a real cross-package edge.
    booted = components["api/client.go::Client.Boot"]
    assert "pkg/workers/worker.go::Spawn" in booted.depends_on


def test_megarepo_cross_module_resolution(tmp_path: Path) -> None:
    """Two sibling Go modules (the submodule layout): imports across them
    are project-internal because every go.mod under the scan root counts."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "go.mod").write_text(
        "module github.com/example/alpha\n\ngo 1.22\n", encoding="utf-8"
    )
    (tmp_path / "alpha" / "svc").mkdir()
    (tmp_path / "alpha" / "svc" / "svc.go").write_text(
        """\
package svc

func Serve() error { return nil }
""",
        encoding="utf-8",
    )
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "go.mod").write_text(
        "module github.com/example/beta\n\ngo 1.22\n", encoding="utf-8"
    )
    (tmp_path / "beta" / "main.go").write_text(
        """\
package main

import "github.com/example/alpha/svc"

type App struct{}

func (a *App) Run() error {
	return svc.Serve()
}
""",
        encoding="utf-8",
    )

    components = DependencyParser(str(tmp_path)).parse_repository()
    run = components["beta/main.go::App.Run"]
    assert "alpha/svc/svc.go::Serve" in run.depends_on
