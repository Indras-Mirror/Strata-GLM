#!/usr/bin/env bash
# glm-engine.sh - the GLM engine as Strata's server starts it (`<exe> --serve <args>`), with memguard's protections
# in the FOREGROUND (memguard.sh backgrounds its command, which would give it /dev/null as stdin and break the
# stdin/stdout protocol): ComfyUI check, the shared GPU lock (held while the server runs), a 84 GiB RAM cap, no swap.
set -u
cd "$(dirname "$0")/../../.."
if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "glm-engine: refusing - ComfyUI is running on :8188 (RAM)" >&2; exit 3
fi
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
export GGML_NO_BACKTRACE=1
exec flock "$LOCK" systemd-run --user --scope -q --unit="glm-serve-$$" -p MemoryMax=84G -p MemorySwapMax=0 \
    "$PWD/build-glm-gpu/glm_generate" "$@"
