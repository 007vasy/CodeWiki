"""Tree-sitter based Go analyzer.

Extracts structs, interfaces, package-level functions, and methods as
documentable components, plus embedded-type, field/parameter/result-type,
instantiation, and call dependency edges.

Go specifics the Scala-style analyzer had to adapt to:

- There are no classes. Structs map to ``class`` and interfaces to
  ``interface`` (both leaf-eligible coarse types); methods carry their
  receiver base type, so a method's logical name is ``RecvType.Name``.
- Cross-package references are always qualified by the *file-local* import
  alias (``types.NewID()``). Aliases are collected per file, classified as
  project-internal or external against every ``go.mod`` module under the
  scan root (cached per repo), and calls on external aliases are dropped
  instead of becoming noise edges. Project-internal qualified callees are
  emitted as ``alias.Name``: the global resolver matches them through the
  defining file's ``qualified_name`` (``pkg.Name``, last two parts) when the
  alias equals the package name, and through the bare tail otherwise.
- Receiver, parameter, and local-variable inference (``:=`` composite
  literals, ``var`` declarations, ``new(T)``, chained selectors through
  in-file struct fields) resolves ``s.store.Save()``-shaped calls to
  ``Store.Save``.
"""

import logging
import os
import re
from pathlib import Path

import tree_sitter_go
from tree_sitter import Language, Parser

from codewiki.src.be.dependency_analyzer.models.core import CallRelationship, Node

logger = logging.getLogger(__name__)

# Go builtin types and ubiquitous single-word types that never point at a
# repository component.
GO_PRIMITIVE_TYPES = frozenset(
    {
        "any",
        "bool",
        "byte",
        "comparable",
        "complex64",
        "complex128",
        "error",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "rune",
        "string",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
    }
)

# Builtin functions (always bare calls, never repository components) plus
# the two most collision-prone stdlib interface methods, filtered only when
# a call cannot be resolved to any component.
GO_CORE_CALLS = frozenset(
    {
        "append",
        "cap",
        "clear",
        "close",
        "copy",
        "delete",
        "len",
        "make",
        "max",
        "min",
        "new",
        "panic",
        "print",
        "println",
        "recover",
        "String",
        "Error",
    }
)

# Directories pruned while collecting go.mod module roots.
_PRUNED_DIRS = {".git", "node_modules", "vendor", "testdata"}

_MODULE_DECLARATION_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)

# repo_path -> [(module_root_abs_dir, module_path), ...] longest-path first.
_GO_MODULE_CACHE: dict[str, list[tuple[str, str]]] = {}


def _collect_go_modules(repo_path: str) -> list[tuple[str, str]]:
    """Find every ``go.mod`` under *repo_path* and its module path.

    Cached per repo: the sweep calls the analyzer once per file, so the walk
    must happen a single time. Longest module paths first so nested modules
    win prefix matching.
    """
    if not repo_path:
        return []
    cached = _GO_MODULE_CACHE.get(repo_path)
    if cached is not None:
        return cached

    modules: list[tuple[str, str]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(repo_path):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _PRUNED_DIRS and not d.startswith(".")
            )
            if "go.mod" not in filenames:
                continue
            try:
                with open(os.path.join(dirpath, "go.mod"), encoding="utf-8") as fh:
                    match = _MODULE_DECLARATION_RE.search(fh.read())
            except OSError:
                continue
            if match:
                modules.append((dirpath, match.group(1)))
    except OSError as e:  # noqa: BLE001 — module discovery is best-effort
        logger.debug("go.mod discovery failed under %s: %s", repo_path, e)
        return []

    modules.sort(key=lambda m: len(m[1]), reverse=True)
    _GO_MODULE_CACHE[repo_path] = modules
    return modules


class TreeSitterGoAnalyzer:
    def __init__(self, file_path: str, content: str, repo_path: str | None = None):
        self.file_path = Path(file_path)
        self.content = content
        self.repo_path = repo_path or ""
        self.nodes: list[Node] = []
        self.call_relationships: list[CallRelationship] = []
        # Same-file symbol table keyed by logical name ("Foo", "Foo.Bar").
        self.top_level_nodes: dict = {}
        # In-file struct field types: "Foo" -> {"field": "Type"}.
        self.struct_fields: dict[str, dict[str, str]] = {}
        # Import alias -> import path, from the file's import block.
        self.import_aliases: dict[str, str] = {}
        # The file's own package clause name.
        self.package_name = ""
        self.go_modules: list[tuple[str, str]] = []
        self.seen_relationships: set = set()
        self._analyze()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_relative_path(self) -> str:
        if self.repo_path:
            try:
                return os.path.relpath(str(self.file_path), self.repo_path)
            except ValueError:
                return str(self.file_path)
        return str(self.file_path)

    def _get_component_id(self, logical_name: str) -> str:
        return f"{self._get_relative_path()}::{logical_name}"

    def _analyze(self):
        try:
            go_language = Language(tree_sitter_go.language())
            parser = Parser(go_language)
            tree = parser.parse(bytes(self.content, "utf8"))
            root = tree.root_node
            lines = self.content.splitlines()

            self.go_modules = _collect_go_modules(self.repo_path)
            self._collect_package_and_imports(root)

            self._extract_nodes(root, lines)
            self._extract_relationships(root)
        except Exception as e:  # noqa: BLE001 — a broken file must not abort the sweep
            logger.error(f"Error parsing Go file {self.file_path}: {e}")

    @staticmethod
    def _field_text(node, field_name: str) -> str | None:
        child = node.child_by_field_name(field_name)
        return child.text.decode() if child is not None else None

    @staticmethod
    def _strip_quotes(literal: str) -> str:
        if len(literal) >= 2 and literal[0] in ('"', "`") and literal[-1] == literal[0]:
            return literal[1:-1]
        return literal

    def _collect_package_and_imports(self, root):
        for child in root.children:
            if child.type == "package_clause":
                # package_identifier is a plain named child, not field-named.
                for sub in child.children:
                    if sub.type == "package_identifier":
                        self.package_name = sub.text.decode()
                        break
            elif child.type == "import_declaration":
                for spec in child.children:
                    if spec.type == "import_spec_list":
                        specs = [
                            s for s in spec.children if s.type == "import_spec"
                        ]
                    elif spec.type == "import_spec":
                        specs = [spec]
                    else:
                        continue
                    for s in specs:
                        path_text = self._field_text(s, "path")
                        if not path_text:
                            continue
                        path = self._strip_quotes(path_text)
                        alias_node = s.child_by_field_name("name")
                        alias = (
                            alias_node.text.decode()
                            if alias_node is not None
                            else path.rstrip("/").rsplit("/", 1)[-1]
                        )
                        if alias and alias != "_":
                            self.import_aliases[alias] = path

    def _is_project_import(self, import_path: str) -> bool:
        """True when *import_path* is covered by a go.mod module under the
        scan root (this is what makes a megarepo of separate Go modules
        resolve cross-repo imports as project edges)."""
        for module_root, module_path in self.go_modules:
            if import_path == module_path or import_path.startswith(
                module_path + "/"
            ):
                return os.path.isdir(
                    os.path.join(
                        module_root, import_path[len(module_path):].lstrip("/")
                    )
                ) or import_path == module_path
        return False

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _find_containing_function(self, node):
        """Nearest enclosing function_declaration/method_declaration,
        including *node* itself (signature edges pass the function node)."""
        current = node
        while current is not None:
            if current.type in ("function_declaration", "method_declaration"):
                return current
            current = current.parent
        return None

    def _receiver_type_of(self, func_node) -> str | None:
        receiver = func_node.child_by_field_name("receiver")
        if receiver is None:
            return None
        for child in receiver.children:
            if child.type == "parameter_declaration":
                return self._base_type_name(child.child_by_field_name("type"))
        return None

    def _resolve_caller(self, node) -> str | None:
        func = self._find_containing_function(node)
        if func is not None:
            if func.type == "method_declaration":
                recv = self._receiver_type_of(func)
                name = self._field_text(func, "name")
                if recv and name:
                    return self._get_component_id(f"{recv}.{name}")
            name = self._field_text(func, "name")
            if name:
                return self._get_component_id(name)
        return None

    def _enclosing_type_name(self, node) -> str | None:
        """Nearest enclosing type_spec (struct/interface) name, if the node
        sits directly in one (fields, interface elements)."""
        current = node.parent
        while current is not None:
            if current.type == "type_spec":
                return self._field_text(current, "name")
            if current.type in ("function_declaration", "method_declaration"):
                return None
            current = current.parent
        return None

    # ------------------------------------------------------------------
    # Type-name extraction
    # ------------------------------------------------------------------

    def _base_type_name(self, type_node) -> str | None:
        """Primary named type of a type node, stripping pointers and
        generics (``*Server``, ``Server[T]`` -> ``Server``)."""
        if type_node is None:
            return None
        if type_node.type == "type_identifier":
            return type_node.text.decode()
        if type_node.type == "qualified_type":
            name_node = type_node.child_by_field_name("name")
            return name_node.text.decode() if name_node is not None else None
        if type_node.type in ("pointer_type", "parenthesized_type", "generic_type"):
            inner = type_node.child_by_field_name("type")
            if inner is None:
                # pointer/generic inner types are plain named children, not
                # field-named ones (``*Server`` -> type_identifier Server).
                for child in type_node.children:
                    if child.is_named:
                        inner = child
                        break
            return self._base_type_name(inner)
        return None

    @staticmethod
    def _struct_field_list(struct_type):
        """The field_declaration_list child of a struct_type (it is a plain
        named child, not a field-named one)."""
        for child in struct_type.children:
            if child.type == "field_declaration_list":
                return child
        return None

    def _named_type_names(self, type_node) -> list[str]:
        """Every named type nested inside a type node, descending pointers,
        slices, arrays, maps (value type), channels, and generics. Structured
        composites (struct/interface literals) are skipped."""
        if type_node is None:
            return []
        t = type_node.type
        if t == "type_identifier":
            return [type_node.text.decode()]
        if t == "generic_type":
            base = type_node.child_by_field_name("type")
            if base is None:
                for child in type_node.children:
                    if child.is_named:
                        base = child
                        break
            return self._named_type_names(base) if base is not None else []
        if t in ("pointer_type", "parenthesized_type", "slice_type", "array_type"):
            inner = None
            for field in ("type", "element"):
                inner = type_node.child_by_field_name(field)
                if inner is not None:
                    break
            if inner is None:
                for child in type_node.children:
                    if child.is_named:
                        inner = child
                        break
            return self._named_type_names(inner) if inner is not None else []
        if t == "map_type":
            value = type_node.child_by_field_name("type")
            return self._named_type_names(value) if value is not None else []
        if t == "channel_type":
            value = type_node.child_by_field_name("value")
            if value is None:
                for child in type_node.children:
                    if child.type.endswith("type"):
                        value = child
                        break
            return self._named_type_names(value) if value is not None else []
        return []

    # ------------------------------------------------------------------
    # Pass 1: components
    # ------------------------------------------------------------------

    def _extract_nodes(self, node, lines):
        if node.type == "type_spec":
            name = self._field_text(node, "name")
            type_node = node.child_by_field_name("type")
            if name and type_node is not None:
                if type_node.type == "struct_type":
                    self._add_node(
                        node,
                        name,
                        "class",
                        "struct",
                        lines,
                        base_classes=self._struct_embedded_types(type_node),
                        qualified_name=self._qualified(name),
                        anchor=node.parent,
                    )
                    self.struct_fields[name] = self._struct_field_types(type_node)
                elif type_node.type == "interface_type":
                    self._add_node(
                        node,
                        name,
                        "interface",
                        "interface",
                        lines,
                        base_classes=self._interface_embedded_types(type_node),
                        qualified_name=self._qualified(name),
                        anchor=node.parent,
                    )
            # Non-struct/interface type specs and aliases stay out: they
            # are rarely doc-worthy units and mostly noise (``type ID = string``).

        elif node.type == "function_declaration":
            name = self._field_text(node, "name")
            if name:
                self._add_node(
                    node,
                    name,
                    "function",
                    "function",
                    lines,
                    parameters=self._extract_parameters(node),
                    qualified_name=self._qualified(name),
                )

        elif node.type == "method_declaration":
            name = self._field_text(node, "name")
            recv = self._receiver_type_of(node)
            if name and recv:
                self._add_node(
                    node,
                    f"{recv}.{name}",
                    "method",
                    "method",
                    lines,
                    class_name=recv,
                    parameters=self._extract_parameters(node),
                    qualified_name=self._qualified(f"{recv}.{name}"),
                )

        for child in node.children:
            self._extract_nodes(child, lines)

    def _qualified(self, logical_name: str) -> str:
        if self.package_name:
            return f"{self.package_name}.{logical_name}"
        return logical_name

    def _extract_parameters(self, func_node) -> list[str] | None:
        params_node = func_node.child_by_field_name("parameters")
        if params_node is None:
            return None
        params = []
        for child in params_node.children:
            if child.type in ("parameter_declaration", "variadic_parameter_declaration"):
                params.append(child.text.decode().strip())
        return params or None

    def _struct_embedded_types(self, struct_type) -> list[str] | None:
        field_list = self._struct_field_list(struct_type)
        if field_list is None:
            return None
        embedded = []
        for field in field_list.children:
            if field.type != "field_declaration":
                continue
            if field.child_by_field_name("name") is not None:
                continue
            type_node = field.child_by_field_name("type")
            name = self._base_type_name(type_node)
            if name:
                embedded.append(name)
        return embedded or None

    def _struct_field_types(self, struct_type) -> dict[str, str]:
        field_list = self._struct_field_list(struct_type)
        if field_list is None:
            return {}
        fields: dict[str, str] = {}
        for field in field_list.children:
            if field.type != "field_declaration":
                continue
            name_node = field.child_by_field_name("name")
            type_node = field.child_by_field_name("type")
            if name_node is None or type_node is None:
                continue
            name = self._base_type_name_from_field_name(name_node)
            candidates = self._named_type_names(type_node)
            if name and candidates:
                fields[name] = candidates[0]
        return fields

    @staticmethod
    def _base_type_name_from_field_name(name_node) -> str | None:
        if name_node is None:
            return None
        if name_node.type in ("field_identifier", "identifier"):
            return name_node.text.decode()
        return None

    def _interface_embedded_types(self, interface_type) -> list[str] | None:
        embedded = []
        for child in interface_type.children:
            if child.type != "type_elem":
                continue
            for sub in child.children:
                name = self._base_type_name(sub)
                if name:
                    embedded.append(name)
        return embedded or None

    def _add_node(
        self,
        node,
        logical_name: str,
        component_type: str,
        node_type: str,
        lines,
        class_name: str | None = None,
        parameters: list[str] | None = None,
        base_classes: list[str] | None = None,
        qualified_name: str | None = None,
        anchor=None,
    ):
        component_id = self._get_component_id(logical_name)
        relative_path = self._get_relative_path()

        docstring = ""
        # Doc comments precede the declaration statement: the type_spec for
        # funcs/methods, but the enclosing type_declaration for types.
        comment = (anchor if anchor is not None else node).prev_sibling
        comment_parts = []
        while comment is not None and comment.type == "comment":
            comment_parts.append(comment.text.decode().lstrip("/").strip())
            comment = comment.prev_sibling
        if comment_parts:
            docstring = "\n".join(reversed(comment_parts)).strip()

        start_line_idx = node.start_point[0]
        end_line_idx = node.end_point[0] + 1
        code_snippet = (
            "\n".join(lines[start_line_idx:end_line_idx]) if start_line_idx < len(lines) else ""
        )

        display_name = f"{node_type} {logical_name}"

        node_obj = Node(
            id=component_id,
            name=logical_name,
            component_type=component_type,
            file_path=str(self.file_path),
            relative_path=relative_path,
            source_code=code_snippet,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            has_docstring=bool(docstring),
            docstring=docstring,
            parameters=parameters,
            node_type=node_type,
            base_classes=base_classes,
            class_name=class_name,
            display_name=display_name,
            component_id=component_id,
            language="go",
            qualified_name=qualified_name,
        )
        self.nodes.append(node_obj)
        self.top_level_nodes[logical_name] = node_obj

    # ------------------------------------------------------------------
    # Pass 2: relationships
    # ------------------------------------------------------------------

    def _extract_relationships(self, node):
        if node.type == "type_spec":
            type_node = node.child_by_field_name("type")
            name = self._field_text(node, "name")
            if name and type_node is not None:
                caller_id = self._get_component_id(name)
                if type_node.type == "struct_type":
                    self._emit_struct_field_edges(caller_id, type_node)
                    self._emit_embedded_edges(caller_id, self._struct_embedded_types(type_node), node)
                elif type_node.type == "interface_type":
                    self._emit_embedded_edges(caller_id, self._interface_embedded_types(type_node), node)
        elif node.type in ("function_declaration", "method_declaration"):
            self._emit_signature_type_edges(node)
        elif node.type == "call_expression":
            self._emit_call_edge(node)
        elif node.type == "composite_literal":
            self._emit_composite_literal_edge(node)

        for child in node.children:
            self._extract_relationships(child)

    def _emit_signature_type_edges(self, func_node):
        caller_id = self._resolve_caller(func_node)
        if not caller_id:
            return
        params_node = func_node.child_by_field_name("parameters")
        if params_node is not None:
            for child in params_node.children:
                if child.type in ("parameter_declaration", "variadic_parameter_declaration"):
                    type_node = child.child_by_field_name("type")
                    if type_node is not None:
                        self._emit_type_edge(caller_id, type_node, child.start_point[0] + 1)
        result_node = func_node.child_by_field_name("result")
        if result_node is not None:
            self._emit_type_edge(caller_id, result_node, func_node.start_point[0] + 1)

    def _emit_struct_field_edges(self, caller_id: str, struct_type):
        field_list = self._struct_field_list(struct_type)
        if field_list is None:
            return
        for field in field_list.children:
            if field.type != "field_declaration":
                continue
            type_node = field.child_by_field_name("type")
            if type_node is not None:
                self._emit_type_edge(caller_id, type_node, field.start_point[0] + 1)

    def _emit_embedded_edges(self, caller_id: str, embedded: list[str] | None, node):
        if not embedded:
            return
        call_line = node.start_point[0] + 1
        for type_name in embedded:
            if self._is_primitive(type_name):
                continue
            self._add_type_relationship(caller_id, type_name, call_line)

    def _emit_composite_literal_edge(self, composite_node):
        caller_id = self._resolve_caller(composite_node)
        if not caller_id:
            return
        type_node = composite_node.child_by_field_name("type")
        if type_node is None:
            return
        self._emit_type_edge(caller_id, type_node, composite_node.start_point[0] + 1)

    def _emit_type_edge(self, caller_id: str, type_node, call_line: int):
        """Emit component edges for every named type a signature/field/
        composite references. Qualified types (``pkg.T``) resolve through the
        file's import table: external packages are dropped, project packages
        contribute the bare type name for global resolution."""
        if type_node is None:
            return
        if type_node.type == "qualified_type":
            package_node = type_node.child_by_field_name("package")
            name_node = type_node.child_by_field_name("name")
            if package_node is None or name_node is None:
                return
            package = package_node.text.decode()
            type_name = name_node.text.decode()
            if self._is_primitive(type_name):
                return
            import_path = self.import_aliases.get(package)
            if import_path is not None and not self._is_project_import(import_path):
                return
            self._add_type_relationship(caller_id, type_name, call_line)
            return

        for type_name in self._named_type_names(type_node):
            if self._is_primitive(type_name):
                continue
            self._add_type_relationship(caller_id, type_name, call_line)

    def _emit_call_edge(self, call_node):
        caller_id = self._resolve_caller(call_node)
        if not caller_id:
            return
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            return
        call_line = call_node.start_point[0] + 1

        if func_node.type == "identifier":
            self._emit_bare_call_edge(caller_id, func_node.text.decode(), call_node, call_line)
        elif func_node.type == "selector_expression":
            self._emit_selector_call_edge(caller_id, func_node, call_node, call_line)

    def _emit_bare_call_edge(self, caller_id: str, callee_name: str, call_node, call_line: int):
        if callee_name == "new":
            args = call_node.child_by_field_name("arguments")
            if args is not None:
                for child in args.children:
                    if child.type.endswith("type") or child.type == "type_identifier":
                        self._emit_type_edge(caller_id, child, call_line)
                    break
            return
        if callee_name in GO_CORE_CALLS:
            return
        func = self._find_containing_function(call_node)
        candidates = [callee_name]
        if func is not None and func.type == "method_declaration":
            recv = self._receiver_type_of(func)
            if recv:
                candidates.insert(0, f"{recv}.{callee_name}")
        resolved_id = self._resolve_candidates(candidates)
        if resolved_id:
            self._add_relationship_raw(caller_id, resolved_id, call_line, True)
        elif not self._is_call_noise(callee_name):
            self._add_relationship_raw(caller_id, callee_name, call_line, False)

    def _emit_selector_call_edge(self, caller_id: str, func_node, call_node, call_line: int):
        operand = func_node.child_by_field_name("operand")
        field_node = func_node.child_by_field_name("field")
        if operand is None or field_node is None:
            return
        method_name = field_node.text.decode()

        if operand.type == "identifier":
            operand_name = operand.text.decode()
            import_path = self.import_aliases.get(operand_name)
            if import_path is not None:
                if self._is_project_import(import_path):
                    # Project-internal package call: ``alias.Name``. The
                    # global resolver matches the defining file's
                    # ``pkg.Name`` qualified name, or falls back to the
                    # bare tail when the alias is custom.
                    self._add_relationship_raw(
                        caller_id, f"{operand_name}.{method_name}", call_line, False
                    )
                # Calls on external packages are dropped, not unresolved.
                return
            var_type = self._find_variable_type(call_node, operand_name)
            if var_type:
                resolved_id = self._resolve_candidates([f"{var_type}.{method_name}"])
                if resolved_id:
                    self._add_relationship_raw(caller_id, resolved_id, call_line, True)
                    return
                if not self._is_call_noise(method_name):
                    self._add_relationship_raw(
                        caller_id, f"{var_type}.{method_name}", call_line, False
                    )
                return
            if not self._is_call_noise(method_name):
                self._add_relationship_raw(caller_id, method_name, call_line, False)
            return

        if operand.type == "selector_expression":
            var_type = self._resolve_selector_operand(operand)
            if var_type:
                resolved_id = self._resolve_candidates([f"{var_type}.{method_name}"])
                if resolved_id:
                    self._add_relationship_raw(caller_id, resolved_id, call_line, True)
                    return
                if not self._is_call_noise(method_name):
                    self._add_relationship_raw(
                        caller_id, f"{var_type}.{method_name}", call_line, False
                    )
                return
            if not self._is_call_noise(method_name):
                self._add_relationship_raw(caller_id, method_name, call_line, False)
            return

        # Composite receiver (call chain, parenthesized, ...): keep the bare
        # method name when it is not stdlib noise.
        if not self._is_call_noise(method_name):
            self._add_relationship_raw(caller_id, method_name, call_line, False)

    def _resolve_selector_operand(self, selector_node) -> str | None:
        """Type of a nested selector like ``s.store``: resolve the base
        variable, then follow in-file struct fields one hop at a time."""
        operand = selector_node.child_by_field_name("operand")
        field_node = selector_node.child_by_field_name("field")
        if operand is None or field_node is None:
            return None
        if operand.type == "identifier":
            base_type = self._find_variable_type(selector_node, operand.text.decode())
        elif operand.type == "selector_expression":
            base_type = self._resolve_selector_operand(operand)
        else:
            return None
        if not base_type:
            return None
        field_name = field_node.text.decode()
        field_type = self.struct_fields.get(base_type, {}).get(field_name)
        return field_type

    def _find_variable_type(self, node, variable_name: str) -> str | None:
        """Best-effort type of a local variable: the enclosing method's
        receiver, its parameters, or a preceding local declaration
        (``var x T``, ``x := T{}``, ``x := &T{}``, ``x := new(T)``)."""
        func = self._find_containing_function(node)
        if func is not None:
            recv = self._receiver_type_of(func)
            receiver_node = func.child_by_field_name("receiver")
            if recv and receiver_node is not None:
                for child in receiver_node.children:
                    if child.type == "parameter_declaration":
                        name_node = child.child_by_field_name("name")
                        if (
                            name_node is not None
                            and name_node.text.decode() == variable_name
                        ):
                            return recv

            params_node = func.child_by_field_name("parameters")
            if params_node is not None:
                for child in params_node.children:
                    if child.type not in ("parameter_declaration", "variadic_parameter_declaration"):
                        continue
                    name_node = child.child_by_field_name("name")
                    if name_node is None or name_node.text.decode() != variable_name:
                        continue
                    type_node = child.child_by_field_name("type")
                    if type_node is not None:
                        name = self._base_type_name(type_node)
                        if name:
                            return name

            body = func.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    if child.start_byte >= node.start_byte:
                        break
                    inferred = self._declared_type_in(child, variable_name)
                    if inferred:
                        return inferred
        return None

    def _declared_type_in(self, statement, variable_name: str) -> str | None:
        if statement.type == "var_declaration":
            for child in statement.children:
                if child.type == "var_spec":
                    name_node = child.child_by_field_name("name")
                    if name_node is None or name_node.text.decode() != variable_name:
                        continue
                    type_node = child.child_by_field_name("type")
                    if type_node is not None:
                        return self._base_type_name(type_node)
            return None
        if statement.type == "short_var_declaration":
            left = statement.child_by_field_name("left")
            right = statement.child_by_field_name("right")
            if left is None or right is None:
                return None
            names = [
                c.text.decode()
                for c in left.children
                if c.type == "identifier"
            ]
            if variable_name not in names:
                return None
            for expr in right.children:
                type_node = None
                if expr.type == "composite_literal":
                    type_node = expr.child_by_field_name("type")
                elif expr.type == "unary_expression":
                    inner = None
                    for child in expr.children:
                        if child.type == "composite_literal":
                            inner = child
                            break
                    if inner is not None:
                        type_node = inner.child_by_field_name("type")
                elif expr.type == "call_expression":
                    callee = expr.child_by_field_name("function")
                    if callee is not None and callee.type == "identifier" and callee.text.decode() == "new":
                        args = expr.child_by_field_name("arguments")
                        if args is not None:
                            for arg in args.children:
                                if arg.type in ("type_identifier", "pointer_type", "qualified_type", "generic_type"):
                                    type_node = arg
                                break
                if type_node is not None:
                    if type_node.type == "qualified_type":
                        name_node = type_node.child_by_field_name("name")
                        return name_node.text.decode() if name_node is not None else None
                    return self._base_type_name(type_node)
                break
        return None

    # ------------------------------------------------------------------
    # Resolution / noise filtering
    # ------------------------------------------------------------------

    def _resolve_candidates(self, candidates: list[str]) -> str | None:
        for candidate in candidates:
            node_obj = self.top_level_nodes.get(candidate)
            if node_obj is not None:
                return node_obj.id
        return None

    def _add_type_relationship(self, caller_id: str, type_name: str, call_line: int):
        resolved_id = self._resolve_candidates([type_name])
        if resolved_id:
            self._add_relationship_raw(caller_id, resolved_id, call_line, True)
        else:
            self._add_relationship_raw(caller_id, type_name, call_line, False)

    def _add_relationship_raw(
        self, caller: str, callee: str, call_line: int | None, resolved: bool
    ):
        key = (caller, callee, call_line)
        if caller == callee or key in self.seen_relationships:
            return
        self.seen_relationships.add(key)
        self.call_relationships.append(
            CallRelationship(
                caller=caller, callee=callee, call_line=call_line, is_resolved=resolved
            )
        )

    def _is_primitive(self, type_name: str) -> bool:
        return type_name in GO_PRIMITIVE_TYPES

    def _is_call_noise(self, name: str) -> bool:
        return name in GO_CORE_CALLS


def analyze_go_file(
    file_path: str, content: str, repo_path: str | None = None
) -> tuple[list[Node], list[CallRelationship]]:
    analyzer = TreeSitterGoAnalyzer(file_path, content, repo_path)
    return analyzer.nodes, analyzer.call_relationships
