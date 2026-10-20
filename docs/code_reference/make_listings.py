"""
Extract the source of every function, class and method of the pyXRT modules
into small files that the code reference (pyxrt_code_reference.tex) includes.

    python make_listings.py          # from this folder

For each item it writes
    listings/<module>/<qualified name>.py    the code, as in the module
    listings/<module>/<qualified name>.tex   \\lstinputlisting with the true
                                             first line number
and listings/source_version.tex with the commit the listings were taken from.
A class listing holds its header, docstring and fields (up to the first
method); each method has its own listing.  Files that are not valid Python 3
(LiveFourierTransform.py) are listed whole.

Finally it reports items that the .tex files never include, so that the
reference stays complete when the code grows.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "listings"

MODULES = ["bliss_hdf5_reader", "xrf_beamline", "xrf_matrix", "xrf_fit", "xrf_calibration",
           "reduce_image_series", "make_dynamic_mosaic", "fluo_map_fit", "spec_reader",
           "xray_tools", "LiveFourierTransform"]


def _start(node: ast.AST) -> int:
    """First line of a definition, decorators included."""
    decos = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decos])


def _items(tree: ast.Module, n_lines: int):
    """(qualified name, first line, last line) of the module header, of every
    top-level function and class, of every method, and of the module tail.

    Statements between two definitions (constants, section comments) belong to
    the listing of the next definition; the header runs from the top of the
    file to the last statement before the first definition; the tail holds
    what follows the last definition (e.g. ``if __name__ == "__main__":``).
    A class item stops before its first method.
    """
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not defs:
        return
    before = [n for n in tree.body if n.end_lineno < _start(defs[0])]
    head_end = before[-1].end_lineno if before else 0
    if head_end:
        yield "_header", 1, head_end
    prev_end = head_end
    for node in defs:
        first = prev_end + 1
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            end = (_start(methods[0]) - 1) if methods else node.end_lineno
            yield node.name, first, end
            for m in methods:
                yield f"{node.name}.{m.name}", _start(m), m.end_lineno
        else:
            yield node.name, first, node.end_lineno
        prev_end = node.end_lineno
    if prev_end < n_lines:
        yield "_tail", prev_end + 1, n_lines


def _write(module: str, name: str, lines: list[str], first: int, last: int, gobble: int = 0) -> None:
    # trim blank lines at both ends
    while last > first and not lines[last - 1].strip():
        last -= 1
    while first < last and not lines[first - 1].strip():
        first += 1
    if not "".join(lines[first - 1:last]).strip():
        return
    d = OUT / module
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.py").write_text("".join(lines[first - 1:last]))
    caption = f"{module}.py, lines {first}--{last}".replace("_", r"\_")
    (d / f"{name}.tex").write_text(
        f"\\lstinputlisting[firstnumber={first},gobble={gobble},title={{\\listingtitle{{{caption}}}}}]"
        f"{{listings/{module}/{name}.py}}\n")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    rev = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h (%ad)", "--date=short"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
    (OUT / "source_version.tex").write_text(f"\\newcommand{{\\sourceversion}}{{{rev}}}\n")

    produced = []
    for module in MODULES:
        path = REPO / f"{module}.py"
        text = path.read_text()
        lines = text.splitlines(keepends=True)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            _write(module, "_file", lines, 1, len(lines))
            produced.append((module, "_file"))
            continue
        _write(module, "_file", lines, 1, len(lines))
        for name, first, last in _items(tree, len(lines)):
            indent = len(lines[first - 1]) - len(lines[first - 1].lstrip()) if "." in name else 0
            _write(module, name, lines, first, last, gobble=indent)
            produced.append((module, name))

    # coverage: items never included by the document
    used = set()
    for tex in HERE.glob("*.tex"):
        used |= set(re.findall(r"\\pycode\{([^}]*)\}\{([^}]*)\}", tex.read_text()))
    for tex in (HERE / "sections").glob("*.tex"):
        used |= set(re.findall(r"\\pycode\{([^}]*)\}\{([^}]*)\}", tex.read_text()))
    used = {(m.replace("\\_", "_"), n.replace("\\_", "_")) for m, n in used}
    missing = [f"{m}.{n}" for m, n in produced if (m, n) not in used
               and n not in ("_header", "_tail") and (m, "_file") not in used]
    print(f"{len(produced)} listings written to {OUT.relative_to(HERE)}/ (source {rev})")
    if missing:
        print(f"{len(missing)} items not included in the document:")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
