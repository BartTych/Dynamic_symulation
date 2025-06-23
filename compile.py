import sys
import subprocess
from pathlib import Path
import platform

# Parse arguments
debug = "--debug" in sys.argv
args = [arg for arg in sys.argv[1:] if arg != "--debug"]

if not args:
    print("Usage: python compile.py <file.cpp> [--debug]")
    sys.exit(1)

cpp_file = Path(args[0])
output_file = cpp_file.with_suffix(".so")

# Platform-specific setup
system = platform.system()
if system == "Darwin":
    compiler = "clang++"
    python = sys.executable
    extra_flags = ["-undefined", "dynamic_lookup"]
else:
    compiler = "g++"
    python = "python3"
    extra_flags = ["-Wl,--strip-all", "-Wl,--no-undefined"]

# Optimization level
if debug:
    optimization_flags = ["-O0", "-g"]
else:
    optimization_flags = ["-O3"]

# Get pybind11 include paths
pybind_includes = subprocess.check_output(
    [python, "-m", "pybind11", "--includes"], text=True
).strip().split()

# Full compiler command
cmd = [
    compiler,
    *optimization_flags,
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-I/opt/homebrew/include/eigen3",
    *extra_flags,
    *pybind_includes,
    str(cpp_file),
    "-o", str(output_file)
]

print("🔧 Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"✅ Built {output_file} ({'debug' if debug else 'release'})")
