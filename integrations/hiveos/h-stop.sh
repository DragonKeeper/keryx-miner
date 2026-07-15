#!/usr/bin/env bash

set -euo pipefail

# HiveOS can invoke stop from a cwd that no longer exists after package updates.
# Move to a safe location so subsequent script logic is unaffected.
cd / || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# This script is executed by HiveOS when stopping the custom miner.

# If Hive launched this miner in a screen session, close that session too so a
# parent shell/wrapper cannot relaunch the binary.
if command -v screen >/dev/null 2>&1; then
	screen -S "miner" -X quit || true
	screen -S "${CUSTOM_NAME}" -X quit || true
fi

pkill -f "${CUSTOM_MINER_DIR}/h-run.sh" || true
pkill -f "screen.*${CUSTOM_MINERBIN}" || true
pkill -f "screen.*${CUSTOM_NAME}" || true

# Kill by the exact installed binary path first, then by common invocation patterns.
pkill -f "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
pkill -x "${CUSTOM_MINERBIN}" || true
pkill -f "./${CUSTOM_MINERBIN}" || true
pkill -f "/${CUSTOM_NAME}/${CUSTOM_MINERBIN}" || true

# Some wrappers ignore TERM briefly; force-stop if still alive.
sleep 1
pkill -9 -f "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
pkill -9 -x "${CUSTOM_MINERBIN}" || true
