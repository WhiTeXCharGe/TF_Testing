#!/usr/bin/env python3
"""
run_java_only.py

A simple launcher that ONLY runs:
    mvn -q -DskipTests exec:java -Dexec.args="EnvConfig.yaml Schedule.yaml"

No looping, no updates, no schedule changes.
Use this to verify if subprocess + MAVEN_OPTS works correctly.
"""

import os
import subprocess
from pathlib import Path

# ---- Locate project root (with pom.xml) ----
here = Path(__file__).resolve()
project_root = None
for p in [here] + list(here.parents):
    if (p / "pom.xml").exists():
        project_root = p
        break

if project_root is None:
    raise SystemExit("Cannot find pom.xml. Run this inside your Maven project.")

print(f"[INFO] Running inside project: {project_root}")

# ---- Maven executable ----
mvn_exe = "mvn.cmd" if os.name == "nt" else "mvn"

# ---- Build command ----
cmd = [
    mvn_exe,
    "-q",
    "-DskipTests",
    "exec:java",
    f"-Dexec.args=src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml"
]

print("[INFO] Running Java solver via Maven:")
print("       " + " ".join(cmd))

# ---- Set heap size for this process ----
env = os.environ.copy()
env["MAVEN_OPTS"] = "-Xms4g -Xmx8g"   # ← adjust if needed
print(f"[INFO] Using MAVEN_OPTS={env['MAVEN_OPTS']}")

# ---- Run ----
try:
    subprocess.run(cmd, cwd=project_root, check=True, env=env)
    print("\n[SUCCESS] Java solver completed successfully.")
except subprocess.CalledProcessError as e:
    print("\n[ERROR] Java solver failed!")
    print(e)
