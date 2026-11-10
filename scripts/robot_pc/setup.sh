#!/usr/bin/env bash
# Robot-PC setup from an extracted bundle (docs/robot_pc.md). NEVER commands a robot; touches nothing outside ROOT.
#
#   mkdir -p ~/workspace/s4d-robot-pc/releases
#   (cd <download dir> && sha256sum -c s4d-robot-pc-<short>.tar.gz.sha256)
#   tar -C ~/workspace/s4d-robot-pc/releases -xzf s4d-robot-pc-<short>.tar.gz
#   bash ~/workspace/s4d-robot-pc/releases/s4d-robot-pc-<short>/setup.sh [--no-realsense]
#
# ROOT = the folder two levels above this bundle (…/s4d-robot-pc). Creates there: venv/ (python3 -m venv --without-pip,
# pip bootstrapped from the bundled wheel, pinned wheels installed OFFLINE with --require-hashes), src -> and assets ->
# symlinks to this release, config/robot_pc.yaml (copied once, never overwritten; a newer template lands next to it
# as robot_pc.yaml.new), secrets/ (0700, empty: the user copies the server token there), logs/, runs/. Then runs the
# self-check. No apt, no sudo, no system Python packages, no device access, no network.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BUNDLE/../.." && pwd)"
PY="${PYTHON:-python3}"
REALSENSE=1
for a in "$@"; do
  case "$a" in
    --no-realsense) REALSENSE=0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
[ "$(basename "$(dirname "$BUNDLE")")" = "releases" ] || { echo "extract the bundle into <root>/releases/ first" >&2; exit 2; }
[ -f "$BUNDLE/BUNDLE.json" ] || { echo "not a bundle directory: $BUNDLE" >&2; exit 2; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "need python 3.12 (Ubuntu 24.04 python3)" >&2; exit 2; }

echo "[setup] root $ROOT, release $BUNDLE"
mkdir -p "$ROOT/config" "$ROOT/logs" "$ROOT/runs" "$ROOT/secrets"
chmod 700 "$ROOT/secrets"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  "$PY" -m venv --without-pip "$ROOT/venv"   # Ubuntu's python3 lacks ensurepip without python3-venv: no apt needed
fi
VPY="$ROOT/venv/bin/python"
PIPWHL="$(ls "$BUNDLE"/wheels/pip-*.whl)"
"$VPY" "$PIPWHL/pip" install --quiet --no-index --find-links "$BUNDLE/wheels" --require-hashes \
  -r "$BUNDLE/wheels/requirements-pip.txt"
"$VPY" -m pip install --quiet --no-index --find-links "$BUNDLE/wheels" --require-hashes \
  -r "$BUNDLE/wheels/requirements.txt"
if [ "$REALSENSE" = 1 ]; then
  "$VPY" -m pip install --quiet --no-index --find-links "$BUNDLE/wheels" --require-hashes \
    -r "$BUNDLE/wheels/requirements-realsense.txt" || echo "[setup] WARNING: pyrealsense2 not installed (replay still works)"
fi
ln -sfn "$BUNDLE/src" "$ROOT/src.new" && mv -T "$ROOT/src.new" "$ROOT/src"
ln -sfn "$BUNDLE/assets" "$ROOT/assets.new" && mv -T "$ROOT/assets.new" "$ROOT/assets"
if [ ! -f "$ROOT/config/robot_pc.yaml" ]; then
  cp "$BUNDLE/src/configs/robot_pc/robot_pc.yaml" "$ROOT/config/robot_pc.yaml"
elif ! cmp -s "$BUNDLE/src/configs/robot_pc/robot_pc.yaml" "$ROOT/config/robot_pc.yaml"; then
  cp "$BUNDLE/src/configs/robot_pc/robot_pc.yaml" "$ROOT/config/robot_pc.yaml.new"
  echo "[setup] kept your config/robot_pc.yaml; the release template is config/robot_pc.yaml.new"
fi
cp "$BUNDLE/BUNDLE.json" "$ROOT/BUNDLE.json"
echo "[setup] self-check"
"$VPY" -I "$ROOT/src/scripts/robot_pc/selfcheck.py" --root "$ROOT" --out "$ROOT/logs/selfcheck.json"
