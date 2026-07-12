#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# This script is executed by HiveOS when stopping the custom miner.
# It is guaranteed to be called whenever HiveOS senses that the custom miner is being upgraded. (I.E. when the url is changed for the custom install url)
# This performs pre-deletion tasks if HiveOS is updating or reinstalling.
# Check if HiveOS is currently executing an update or re-installation.
if pgrep -f "custom-get" >/dev/null; then
    echo "Custom-get process detected. Running pre-deletion tasks..."
    # Preserve model cache across updates/reinstalls when present.
    if [[ -d "$CUSTOM_MINER_DIR/models" ]]; then
        mv "$CUSTOM_MINER_DIR/models" /hive/miners/custom/models || true
    fi
fi

pkill -f "/${MINER_NAME}/${CUSTOM_MINERBIN}" || true
pkill -f "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
pkill -f "./${CUSTOM_MINERBIN}" || true