#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# This script is executed by HiveOS when stopping the custom miner.

pkill -f "/${MINER_NAME}/${CUSTOM_MINERBIN}" || true
pkill -f "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
pkill -f "./${CUSTOM_MINERBIN}" || true