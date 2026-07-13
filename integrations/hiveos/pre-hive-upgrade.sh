#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# Run the current stop hook first so its preservation logic executes before
# any pre-upgrade migration done here.
if [[ -x "$SCRIPT_DIR/h-stop.sh" ]]; then
    "$SCRIPT_DIR/h-stop.sh" || true
fi

if [[ -d "$CUSTOM_MINER_DIR/models" ]]; then
    if [[ ! -d /hive/miners/custom/models ]]; then
        mv "$CUSTOM_MINER_DIR/models" /hive/miners/custom/models || true
    fi
fi

echo "[keryx] HiveOS Pre-upgrade preparation complete."
echo "[keryx] It is now safe to change the Install URL in the HiveOS custom miner config screen and apply changes."