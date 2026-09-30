#!/usr/bin/env python3
"""Verify maintained documentation against the repository's live contracts."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"
LICENSE = ROOT / "LICENSE"
LOCAL_ONLY_DOCUMENTS = {ROOT / "AGENTS.md"}
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
STATUS_PATTERN = re.compile(r"^> Status: (Current|Experimental|Planned)$")
REVIEW_PATTERN = re.compile(r"^> Last reviewed: (\d{4}-\d{2}-\d{2})$")
SCENARIO_PATTERN = re.compile(
    r'constexpr\s+const\s+char\*\s+k\w+Scenario\s*=\s*"([^"]+)"\s*;',
    re.MULTILINE,
)
QUICKSTART_SCENARIOS = (
    "all-hbm",
    "all-hbf",
    "flat",
    "direct-read",
    "hbf-streaming",
)
SCENARIO_DOCUMENTS = (
    ROOT / "README.md",
    DOCS / "project-layout.md",
    DOCS / "reference" / "model.md",
    DOCS / "reference" / "observability.md",
    DOCS / "reference" / "scenarios.md",
)
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}
RETIRED_REFERENCES = (
    (re.compile(r"\bscenario_compare\b"), "retired scenario_compare tree"),
    (
        re.compile(r"\bHBServe\b"),
        "retired frontend name; the repository is ServeLoop",
    ),
    (
        re.compile(r"prepare_frontier_residency_config\.py"),
        "retired Frontier binder name",
    ),
    (
        re.compile(r"compare_summary_results\.py"),
        "retired report command name",
    ),
    (
        re.compile(r"erase_pending_vpn_dependency_ok"),
        "retired physical-probe identifier",
    ),
    (
        re.compile(r"hbf-subarrays-per-plane\s*=\s*16"),
        "obsolete server HBF subarray count",
    ),
    (
        re.compile(r"HBF subarrays\s+16", re.IGNORECASE),
        "obsolete server HBF topology table",
    ),
    (
        re.compile(r"\bsix policies are available\b", re.IGNORECASE),
        "obsolete policy count",
    ),
    (
        re.compile(r"\bfour (?:aligned )?address domains\b", re.IGNORECASE),
        "obsolete heatmap-domain count",
    ),
    (
        re.compile(
            r"does not yet (?:contain|include) a public software license",
            re.IGNORECASE,
        ),
        "obsolete missing-license claim",
    ),
    (
        re.compile(r"until a license is added", re.IGNORECASE),
        "obsolete missing-license qualification",
    ),
)
IMPLEMENTATION_SUFFIXES = {
    ".cfg",
    ".cmake",
    ".cpp",
    ".hpp",
    ".json",
    ".py",
    ".sh",
    ".txt",
}
EXTERNAL_COVERAGE_REFERENCES = {
    "test_llama31_8b_profile_has_exact_full_model_and_128k_boundary",
}


class DocumentationError(RuntimeError):
    """A maintained-document contract is incomplete or stale."""


def maintained_documents() -> list[Path]:
    paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.split("\0")
    return sorted({
        ROOT / name for name in paths if name and (ROOT / name).is_file()
        and (ROOT / name) not in LOCAL_ONLY_DOCUMENTS
        and (name.startswith("docs/") and name.endswith(".md")
             or "/" not in name and name.endswith(".md")
             or Path(name).name == "README.md")
    })


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return (source.parent / target).resolve()


def markdown_links(path: Path) -> list[Path]:
    links: list[Path] = []
    text = path.read_text(encoding="utf-8")
    for match in LINK_PATTERN.finditer(text):
        target = local_target(path, match.group(1))
        if target is not None:
            links.append(target)
    return links


def verify_metadata(path: Path) -> None:
    header = path.read_text(encoding="utf-8").splitlines()[:12]
    if not any(STATUS_PATTERN.fullmatch(line) for line in header):
        raise DocumentationError(f"{path}: missing maintained status")
    review_matches = [
        match
        for line in header
        if (match := REVIEW_PATTERN.fullmatch(line)) is not None
    ]
    if not review_matches:
        raise DocumentationError(f"{path}: missing last-reviewed date")
    reviewed = date.fromisoformat(review_matches[0].group(1))
    if reviewed > date.today():
        raise DocumentationError(f"{path}: last-reviewed date is in the future")


def verify_index_and_links(documents: list[Path]) -> int:
    indexed = set(markdown_links(INDEX))
    expected = {path.resolve() for path in documents if path != INDEX}
    missing_from_index = sorted(expected - indexed)
    if missing_from_index:
        rendered = ", ".join(
            path.relative_to(ROOT).as_posix() for path in missing_from_index
        )
        raise DocumentationError(
            f"docs/README.md does not index maintained document(s): {rendered}"
        )

    checked_links = 0
    for source in documents:
        for target in markdown_links(source):
            checked_links += 1
            if not target.exists():
                raise DocumentationError(
                    f"{source.relative_to(ROOT)}: broken local link to {target}"
                )
    return checked_links


def verify_retired_references(documents: Iterable[Path]) -> int:
    checked = 0
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for pattern, description in RETIRED_REFERENCES:
            checked += 1
            match = pattern.search(text)
            if match is not None:
                line = text.count("\n", 0, match.start()) + 1
                raise DocumentationError(
                    f"{document.relative_to(ROOT)}:{line}: {description}"
                )
    return checked


def verify_license_contract() -> int:
    lines = LICENSE.read_text(encoding="utf-8").splitlines()
    title = next((line.strip() for line in lines if line.strip()), "")
    if not title:
        raise DocumentationError("LICENSE is empty")
    expected_link = f"[{title}](LICENSE)"
    for document in (ROOT / "README.md", ROOT / "CONTRIBUTING.md"):
        if expected_link not in document.read_text(encoding="utf-8"):
            raise DocumentationError(
                f"{document.relative_to(ROOT)} does not identify the current "
                f"repository license: {title}"
            )
    return 2


def require_catalog_entries(
    catalog: Path,
    paths: Iterable[Path],
    relative_to: Path,
) -> int:
    text = catalog.read_text(encoding="utf-8")
    entries = sorted(set(paths))
    missing = [
        path.relative_to(relative_to).as_posix()
        for path in entries
        if path.relative_to(relative_to).as_posix() not in text
    ]
    if missing:
        raise DocumentationError(
            f"{catalog.relative_to(ROOT)} omits file(s): {', '.join(missing)}"
        )
    return len(entries)


def verify_file_catalogs() -> int:
    count = 0
    count += require_catalog_entries(
        ROOT / "cmake" / "README.md",
        (ROOT / "cmake").rglob("*.cmake"),
        ROOT / "cmake",
    )
    config_entries = list((ROOT / "configs").rglob("*.cfg"))
    config_entries.append(ROOT / "configs" / "parameter-provenance.json")
    count += require_catalog_entries(
        ROOT / "configs" / "README.md",
        config_entries,
        ROOT / "configs",
    )
    for relative in ("reports", "workloads", "hbfsim", "examples"):
        directory = ROOT / relative
        count += require_catalog_entries(
            directory / "README.md",
            (
                path
                for path in directory.rglob("*.py")
                if path.name != "__init__.py"
            ),
            directory,
        )
    for relative in ("hbfsim_client",):
        directory = ROOT / relative
        entries = [
            path
            for path in directory.rglob("*.py")
            if path.name not in {"__init__.py", "__main__.py"}
        ]
        entries.extend(directory.rglob("*.json"))
        count += require_catalog_entries(
            directory / "README.md",
            entries,
            directory,
        )
    verification = ROOT / "verification"
    verification_entries: list[Path] = []
    for relative in ("core", "oracles", "gates"):
        verification_entries.extend(
            path
            for path in (verification / relative).iterdir()
            if path.suffix in {".py", ".sh"} and path.name != "__init__.py"
        )
    verification_entries.extend((verification / "probes").glob("*.cpp"))
    count += require_catalog_entries(
        verification / "README.md",
        verification_entries,
        verification,
    )
    external_assets = ROOT / "evidence" / "external" / "assets"
    count += require_catalog_entries(
        external_assets / "README.md",
        (
            path
            for path in external_assets.rglob("*")
            if path.is_file() and path.name != "README.md"
        ),
        external_assets,
    )
    return count


def verify_coverage_references() -> int:
    coverage = (DOCS / "verification" / "coverage.md").read_text(encoding="utf-8")
    references = sorted(set(re.findall(r"`([^`\n]+)`", coverage)))
    simple_references = [
        reference
        for reference in references
        if re.fullmatch(r"[A-Za-z0-9_.-]+", reference)
        and reference not in EXTERNAL_COVERAGE_REFERENCES
    ]
    implementation_roots = (
        ROOT / "CMakeLists.txt",
        ROOT / "cmake",
        ROOT / "configs",
        ROOT / "evidence",
        ROOT / "reports",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "verification",
        ROOT / "workloads",
    )
    source_files: list[Path] = []
    for implementation_root in implementation_roots:
        candidates = (
            [implementation_root]
            if implementation_root.is_file()
            else implementation_root.rglob("*")
        )
        source_files.extend(
            path
            for path in candidates
            if path.is_file()
            and path.suffix in IMPLEMENTATION_SUFFIXES
            and path != Path(__file__).resolve()
        )
    implementation = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(set(source_files))
    )
    missing = [
        reference for reference in simple_references if reference not in implementation
    ]
    if missing:
        raise DocumentationError(
            "docs/verification/coverage.md has unresolvable reference(s): "
            + ", ".join(missing)
        )
    return len(simple_references)


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise DocumentationError(f"{path.relative_to(ROOT)}: missing {name}")


def verify_scenario_contract() -> tuple[str, ...]:
    source = (ROOT / "src" / "app" / "reference_runner.cpp").read_text(
        encoding="utf-8"
    )
    scenarios = tuple(dict.fromkeys(SCENARIO_PATTERN.findall(source)))
    if not scenarios:
        raise DocumentationError(
            "src/app/reference_runner.cpp: no canonical reference-policy "
            "constants found"
        )
    count_word = NUMBER_WORDS.get(len(scenarios))
    if count_word is None:
        raise DocumentationError("documentation gate has no word for scenario count")
    for document in SCENARIO_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        missing = [scenario for scenario in scenarios if scenario not in text]
        if missing:
            raise DocumentationError(
                f"{document.relative_to(ROOT)} omits scenario(s): {', '.join(missing)}"
            )
    model_text = (DOCS / "reference" / "model.md").read_text(
        encoding="utf-8"
    ).lower()
    if f"{count_word} reference policies are available" not in model_text:
        raise DocumentationError(
            "docs/reference/model.md: reference-policy count disagrees with "
            "the reference runner"
        )
    return scenarios


def verify_schema_and_topology_contracts() -> tuple[int, int]:
    summary_version = literal_assignment(
        ROOT / "reports" / "address_heatmap.py", "SUMMARY_SCHEMA_VERSION"
    )
    for relative in (
        "README.md",
        "docs/guides/workload-replay.md",
        "docs/reference/evidence-policy.md",
        "docs/reference/model.md",
        "docs/reference/scenarios.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(
            rf"schema.{{0,80}}(?:v{summary_version}\b|version\s+{summary_version}\b)",
            text,
            re.IGNORECASE | re.DOTALL,
        ) is None:
            raise DocumentationError(
                f"{relative}: missing summary schema {summary_version} contract"
            )

    domains = literal_assignment(ROOT / "reports" / "address_heatmap.py", "DOMAIN_ORDER")
    if not isinstance(domains, tuple) or not domains:
        raise DocumentationError(
            "reports/address_heatmap.py: missing the public heatmap domain contract"
        )
    domain_count_word = NUMBER_WORDS.get(len(domains))
    if domain_count_word is None:
        raise DocumentationError("documentation gate has no word for heatmap domain count")
    scenario_text = (DOCS / "reference" / "scenarios.md").read_text(encoding="utf-8")
    missing_domains = [domain for domain in domains if domain not in scenario_text]
    if missing_domains:
        raise DocumentationError(
            "docs/reference/scenarios.md omits heatmap domain(s): "
            + ", ".join(missing_domains)
        )
    replay_text = (DOCS / "guides" / "workload-replay.md").read_text(
        encoding="utf-8"
    )
    domain_count_contracts = (
        (ROOT / "reports" / "address_heatmap.py", f"{domain_count_word} address domains"),
        (DOCS / "guides" / "workload-replay.md", f"{domain_count_word} address domains"),
        (DOCS / "reference" / "evidence-policy.md", f"{domain_count_word} domains"),
        (DOCS / "reference" / "scenarios.md", f"{domain_count_word} domains"),
        (ROOT / "tests" / "README.md", f"{domain_count_word}-domain"),
        (
            ROOT / "src" / "physical" / "address_heatmap.cpp",
            f"all {domain_count_word} address domains",
        ),
    )
    for path, phrase in domain_count_contracts:
        if phrase not in path.read_text(encoding="utf-8").lower():
            raise DocumentationError(
                f"{path.relative_to(ROOT)}: heatmap domain count is stale"
            )

    server_config = (
        ROOT / "configs" / "systems" / "server-hbm128-hbf512.cfg"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"^hbf-speed-grade\s*=\s*(\d+)\s*$", server_config, re.MULTILINE
    )
    if match is None:
        raise DocumentationError(
            "configs/systems/server-hbm128-hbf512.cfg: missing HBF speed grade"
        )
    subarrays = int(match.group(1))
    if f"hbf-speed-grade={subarrays}" not in replay_text:
        raise DocumentationError(
            "docs/guides/workload-replay.md: server HBF speed grade disagrees "
            "with the default config"
        )
    return summary_version, 3 + len(domain_count_contracts)


def run_checked(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DocumentationError(f"{description} timed out") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise DocumentationError(f"{description} failed{suffix}")
    return result


def verify_quickstart(
    simulator: Path,
    scenarios: tuple[str, ...],
    summary_version: int,
) -> int:
    if not simulator.is_file():
        raise DocumentationError(f"simulator does not exist: {simulator}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    selected = ",".join(QUICKSTART_SCENARIOS)
    required_fragments = (
        "python3 -m hbfsim quickstart",
        "python3 -B -m hbserve run",
        "--model ../ServeLoop/models/llama31-8b-w8-kv-bf16.json",
        "--system configs/systems/miniquick/4hbm-4hbf.cfg",
        "--requests ../ServeLoop/examples/quickstart-requests.json",
        "--placement weights-hbf-kv-hbm",
        "--generate-semantic-llm out/quickstart/trace.txt",
        "--llm-tokens 4",
        "--llm-layers 2",
        f"--scenarios {selected}",
        "--summary-json out/quickstart/summary.json",
        "SANITY: PASS",
    )
    missing = [fragment for fragment in required_fragments if fragment not in readme]
    if missing:
        raise DocumentationError(
            "README.md quick start omits: " + ", ".join(missing)
        )
    if re.search(r"^\s*--max-ops 128\s*\\?\s*$", readme, re.MULTILINE):
        raise DocumentationError(
            "README.md quick start uses a prefix that cannot exercise mixed routing"
        )

    help_result = run_checked(
        [str(simulator), "--help"], "hbfsim-reference --help"
    )
    help_text = help_result.stdout + help_result.stderr
    if "--scenarios" not in help_text:
        raise DocumentationError("hbfsim-reference --help omits --scenarios")
    missing_help = [scenario for scenario in scenarios if scenario not in help_text]
    if missing_help:
        raise DocumentationError(
            "hbfsim-reference --help omits scenario(s): "
            + ", ".join(missing_help)
        )
    scenario_count_word = NUMBER_WORDS[len(scenarios)]
    if f"omitted runs all {scenario_count_word}" not in help_text.lower():
        raise DocumentationError(
            "hbfsim-reference --help has a stale scenario count"
        )

    with tempfile.TemporaryDirectory(prefix="hbfsim-documentation-") as temporary:
        output = Path(temporary)
        trace = output / "trace.txt"
        summary = output / "summary.json"
        run_checked(
            [
                str(simulator),
                "--generate-semantic-llm",
                str(trace),
                "--llm-tokens",
                "4",
                "--llm-layers",
                "2",
            ],
            "README quick-start trace generation",
        )
        result = run_checked(
            [
                str(simulator),
                "--config",
                str(ROOT / "configs" / "systems" / "server-hbm128-hbf512.cfg"),
                "--config",
                str(
                    ROOT
                    / "configs"
                    / "policies"
                    / "reference"
                    / "server-hbm128-hbf512.cfg"
                ),
                "--trace",
                str(trace),
                "--scenarios",
                selected,
                "--summary-json",
                str(summary),
            ],
            "README quick-start simulation",
        )
        if "SANITY: PASS" not in result.stdout + result.stderr:
            raise DocumentationError("README quick-start simulation did not pass sanity")
        artifact = json.loads(summary.read_text(encoding="utf-8"))
        schema = artifact.get("schema")
        if schema != {
            "name": "hbfsim.simulation.summary",
            "version": summary_version,
        }:
            raise DocumentationError("README quick start emitted an unexpected schema")
        if artifact.get("sanity") != "PASS":
            raise DocumentationError("README quick-start JSON sanity is not PASS")
        rows = artifact.get("scenarios")
        if not isinstance(rows, list):
            raise DocumentationError("README quick-start JSON has no scenario list")
        names = [row.get("name") for row in rows if isinstance(row, dict)]
        if names != list(QUICKSTART_SCENARIOS):
            raise DocumentationError(
                "README quick-start scenario output does not match its command"
            )
        flat = next((row for row in rows if row.get("name") == "flat"), None)
        if not isinstance(flat, dict) or not (
            flat.get("hbm_user_accesses", 0) > 0
            and flat.get("hbf_user_accesses", 0) > 0
        ):
            raise DocumentationError(
                "README quick start does not exercise both sides of flat routing"
            )
    return len(QUICKSTART_SCENARIOS)


def verify(simulator: Path) -> tuple[int, int, int, int]:
    documents = maintained_documents()
    for document in documents:
        verify_metadata(document)
    link_count = verify_index_and_links(documents)
    retired_checks = verify_retired_references(documents)
    license_checks = verify_license_contract()
    catalog_count = verify_file_catalogs()
    coverage_checks = verify_coverage_references()
    scenarios = verify_scenario_contract()
    summary_version, semantic_checks = verify_schema_and_topology_contracts()
    semantic_checks += verify_quickstart(simulator, scenarios, summary_version)
    semantic_checks += retired_checks
    semantic_checks += license_checks
    semantic_checks += coverage_checks
    return len(documents), link_count, catalog_count, semantic_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", required=True, type=Path)
    args = parser.parse_args()
    try:
        document_count, link_count, catalog_count, semantic_checks = verify(
            args.simulator.resolve()
        )
    except (OSError, ValueError, json.JSONDecodeError, DocumentationError) as error:
        print(f"documentation contract FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "documentation contract PASS: "
        f"{document_count} maintained documents, {link_count} local links, "
        f"{catalog_count} cataloged files, {semantic_checks} semantic checks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
