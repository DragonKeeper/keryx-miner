#!/usr/bin/env bash

cd `dirname $0`

[ -t 1 ] && . colors

. h-manifest.conf

[[ -z $CUSTOM_LOG_BASENAME ]] && echo -e "${RED}No CUSTOM_LOG_BASENAME is set${NOCOLOR}" && exit 1
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}Custom config ${YELLOW}$CUSTOM_CONFIG_FILENAME${RED} is not found${NOCOLOR}" && exit 1

# Expose the miner dir and CUDA runtime libs (cuBLAS etc.) for OPoI inference.
# The binary self-installs cuBLAS on first run and registers its path with ldconfig,
# so this is only a belt-and-suspenders hint for the dynamic loader.
export LD_LIBRARY_PATH="$(dirname $0):${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"

# GNU screen on HiveOS commonly mis-renders 24-bit colors; prefer ANSI palette there.
if [[ -n "${STY:-}" && -z "${KERYX_TRUECOLOR:-}" ]]; then
  export KERYX_TRUECOLOR=0
fi

./$CUSTOM_MINERBIN $(< $CUSTOM_CONFIG_FILENAME) --stats-bind 0.0.0.0 --stats-port "$WEB_PORT" $@
