#!/usr/bin/env python3
"""Compare Rust SF2/KUR/OKK/ZDT6 with the unchanged historical C++ kernels.

Run from any directory with `python3 scripts/kernel_parity.py`. The harness
includes the original source files directly and supplies only minimal MOCS
base classes, so the objective expressions remain the historical kernels.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MOCS = Path(os.environ.get("MOCS_CPP_SOURCE", ROOT.parent / "mocs")).resolve()
FIXTURES = {
    "sf2": ("0.25",),
    "kur": ("-1.0", "0.5", "2.0"),
    "okk": (
        "100",
        "100",
        "150",
        "0",
        "140",
        "150",
        "3",
    ),
    "zdt6": ("0.25", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"),
}


def run(command: list[str], **kwargs: object) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr or str(error)) from error
    return result.stdout


def main() -> None:
    compiler = shutil.which("g++-16") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("a C++ compiler is required (tried g++-16, g++, clang++)")
    for source in ("sf2.cc", "kur.cc", "okk.cc", "zdt6.cc"):
        if not (MOCS / source).is_file():
            raise SystemExit(f"historical source not found: {MOCS / source}")

    with tempfile.TemporaryDirectory(prefix="mocs-kernel-parity-") as scratch:
        tmp = Path(scratch)
        include = tmp / "mocs"
        include.mkdir()
        (include / "realgene.h").write_text(
            """#pragma once
#include <vector>
class RealGene { double value, min, max; public:
  explicit RealGene(double x): value(x), min(x), max(x) {}
  RealGene(double lo, double hi): value(0), min(lo), max(hi) {}
  double getValue() const { return value; }
  void init() { value=min+(max-min)*0.5; }
};
class Genome: public std::vector<void*> { public:
  using std::vector<void*>::vector;
  void init() { for (void* g: *this) static_cast<RealGene*>(g)->init(); }
};
class Individual { Genome genes; std::vector<double> fitness; public:
  explicit Individual(const std::vector<RealGene*>& g): genes(g.begin(), g.end()) {}
  explicit Individual(size_t n): genes(n, nullptr) {}
  Genome& getGenome() { return genes; }
  std::vector<double>& getFitness() { return fitness; }
  void setValue(double) {}
};
"""
        )
        (include / "ffunction.h").write_text(
            """#pragma once
#include <string>
class Params { int n; public: explicit Params(int vars): n(vars) {} int getInt(const char* key, int fallback) { return std::string(key)=="numVars" ? n : fallback; } };
class Loadable { public: virtual ~Loadable() = default; };
class FitnessFunction: public Loadable { public: explicit FitnessFunction(Params&) {} virtual void operator()(Individual&) = 0; };
#ifndef HUGE
#define HUGE 1e300
#endif
"""
        )
        (include / "initiator.h").write_text(
            """#pragma once
#include <mocs/ffunction.h>
class Initiator: public Loadable { public: explicit Initiator(Params&) {} virtual Individual* newIndividual() const = 0; };
"""
        )
        cpp = tmp / "parity.cc"
        cpp.write_text(
            f'''#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
using std::min;
using std::max;
#include "{MOCS / 'sf2.cc'}"
#include "{MOCS / 'kur.cc'}"
#include "{MOCS / 'okk.cc'}"
#include "{MOCS / 'zdt6.cc'}"
'''
            + r'''template<class Kernel> void emit(const char* name, int n, const std::vector<double>& xs) {
  std::vector<RealGene> owned; owned.reserve(xs.size());
  for (double x: xs) owned.emplace_back(x);
  std::vector<RealGene*> ptrs; ptrs.reserve(owned.size());
  for (auto& g: owned) ptrs.push_back(&g);
  Individual ind(ptrs); Params p(n); Kernel kernel(p); kernel(ind);
  std::cout << "CPP_PARITY " << name << " " << std::setprecision(17)
            << ind.getFitness()[0] << " " << ind.getFitness()[1] << "\n";
}
int main() {
  emit<SF2>("sf2", 1, {0.25});
  emit<KUR>("kur", 3, {-1.0, 0.5, 2.0});
  emit<ZDT6>("zdt6", 10, {0.25,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9});
  emit<OKK>("okk", 7, {100,100,150,0,140,150,3});
}
'''
        )
        binary = tmp / "parity"
        run([compiler, "-std=gnu++17", "-O2", "-I", str(tmp), str(cpp), "-o", str(binary)])
        cpp_out = run([str(binary)])
        env = os.environ.copy()
        env["MOCS_KERNEL_PARITY"] = "1"
        rust_out = run(
            ["cargo", "test", "--offline", "--test", "kernel_parity", "--", "--nocapture"],
            cwd=ROOT,
            env=env,
        )

    pattern = re.compile(r"(?:CPP|RUST)_PARITY (\w+) ([^ ]+) ([^\s]+)")
    cpp_values = {name: (float(a), float(b)) for name, a, b in pattern.findall(cpp_out)}
    rust_values = {name: (float(a), float(b)) for name, a, b in pattern.findall(rust_out)}
    if cpp_values.keys() != FIXTURES.keys() or rust_values.keys() != FIXTURES.keys():
        raise SystemExit(f"missing parity records; C++={cpp_values}, Rust={rust_values}")
    for name in FIXTURES:
        for index, (expected, actual) in enumerate(zip(cpp_values[name], rust_values[name])):
            tolerance = 1e-12 + 1e-10 * abs(expected)
            if not math.isfinite(actual) or abs(actual - expected) > tolerance:
                raise SystemExit(
                    f"{name} objective {index}: C++={expected:.17g}, Rust={actual:.17g}, tolerance={tolerance:.3g}"
                )
        print(f"{name}: parity passed; C++={cpp_values[name]}, Rust={rust_values[name]}")


if __name__ == "__main__":
    main()
