#!/usr/bin/env python3
"""Build provenance tracks edited source, not the runtime checkout."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BuildProvenanceTests(unittest.TestCase):
    cmake: str
    compiler: str

    def test_incremental_build_updates_the_immutable_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbfsim-build-provenance-") as directory:
            root = Path(directory)
            source = root / "project"
            build = root / "build"
            (source / "src").mkdir(parents=True)
            (source / "cmake").mkdir()
            shutil.copy2(ROOT / "cmake/BuildProvenance.cmake", source / "cmake")
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.20)\n'
                'project(ProvenanceProbe LANGUAGES CXX)\n'
                'add_custom_target(provenance COMMAND "${CMAKE_COMMAND}"\n'
                '  "-DSOURCE_ROOT=${CMAKE_SOURCE_DIR}"\n'
                '  "-DOUTPUT_DIR=${CMAKE_BINARY_DIR}/generated"\n'
                '  -P "${CMAKE_SOURCE_DIR}/cmake/BuildProvenance.cmake"\n'
                '  BYPRODUCTS "${CMAKE_BINARY_DIR}/generated/hbfsim_build_provenance.hpp"\n'
                '             "${CMAKE_BINARY_DIR}/generated/hbfsim_source_manifest.txt")\n'
                'add_executable(probe src/main.cpp)\n'
                'add_dependencies(probe provenance)\n'
                'target_include_directories(probe PRIVATE "${CMAKE_BINARY_DIR}/generated")\n',
                encoding="utf-8",
            )
            program = source / "src/main.cpp"
            program.write_text(
                '#include "hbfsim_build_provenance.hpp"\n'
                '#include <cstdio>\n'
                'int main() { std::puts(HBFSIM_SOURCE_SHA256); }\n',
                encoding="utf-8",
            )

            def command(*arguments: str) -> str:
                completed = subprocess.run(
                    arguments, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                return completed.stdout.strip()

            command(self.cmake, "-S", str(source), "-B", str(build),
                    f"-DCMAKE_CXX_COMPILER={self.compiler}")
            program.write_text(program.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            command(self.cmake, "--build", str(build))
            executable = build / "probe"
            original = command(str(executable))
            manifest = build / "generated/hbfsim_source_manifest.txt"
            self.assertEqual(original, hashlib.sha256(manifest.read_bytes()).hexdigest())
            original_mtime = executable.stat().st_mtime_ns
            command(self.cmake, "--build", str(build))
            self.assertEqual(executable.stat().st_mtime_ns, original_mtime)
            saved = root / "previous-probe"
            shutil.copy2(executable, saved)
            program.write_text(program.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertEqual(command(str(executable)), original)
            command(self.cmake, "--build", str(build))
            rebuilt = command(str(executable))
            self.assertNotEqual(rebuilt, original)
            self.assertEqual(rebuilt, hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(command(str(saved)), original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--compiler", required=True)
    options = parser.parse_args()
    BuildProvenanceTests.cmake = options.cmake
    BuildProvenanceTests.compiler = options.compiler
    unittest.main(argv=[__file__])
