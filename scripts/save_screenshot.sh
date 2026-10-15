#!/usr/bin/env bash
# Moves the newest .png from your Desktop into screenshots/<name>.png
set -e
name="${1:?give a name, e.g. ci_01_workflow_run}"
latest="$(ls -t "$HOME"/Desktop/*.png 2>/dev/null | head -1)"
if [ -z "$latest" ]; then
  echo "No screenshot found on Desktop. Take one with Cmd+Shift+4 first."; exit 1
fi
mkdir -p screenshots
mv "$latest" "screenshots/${name}.png"
echo "saved screenshots/${name}.png  (from: $(basename "$latest"))"
