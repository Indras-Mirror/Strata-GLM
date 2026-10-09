#!/usr/bin/env bash
# --renorm-skip gate (glm worktree, build-glm-gpu).  Waits for the shared GPU lock AND enough RAM (DS4 calib holds
# both), then runs.  memguard refuses (rc 3) when MemAvailable is short, so each run retries until it starts.
#   A: NON-ABLATED, skip 0.15, renorm OFF vs ON  -> the "no quality regression" check (ppl must not rise)
#   B: ABLATED (--lora-exps) + skip 0.15, renorm OFF vs ON -> does renorm un-break ablated+skip (the s20 loop)?
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 524288 --vram-margin 5.5
        --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt --stop 154820,154827,154829)
SKIP=(--skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.15 --prefill-chunk 1024)

wait_gpu() {   # block until the lock is free and MemAvailable >= 75 GiB
  while :; do
    local a; a=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
    if flock -n "$LOCK" true 2>/dev/null && [ "${a:-0}" -ge 75 ]; then return 0; fi
    sleep 60
  done
}
run() {        # <tag> <extra args...>
  local tag=$1; shift
  while :; do
    wait_gpu
    bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" --ids-file "$D/prompts/eval_code.i32" -n 32 --ppl \
        > "$D/renorm-$tag.log" 2>&1
    local rc=$?
    [ "$rc" -ne 3 ] && break        # rc 3 = memguard refused (RAM short): wait and retry
    sleep 60
  done
  printf '%-8s exit=%s  %s\n' "$tag" "$rc" "$(grep -E '^ppl:' "$D/renorm-$tag.log" | tail -1)" | tee -a "$D/renorm-gate.summary"
}

echo "=== gate-renorm start $(date +%T) (waiting for GPU+RAM) ===" | tee "$D/renorm-gate.summary"
run a-off "${COMMON[@]}" --pcie 0.35                  "${SKIP[@]}" --chunk-mmq
run a-on  "${COMMON[@]}" --pcie 0.35 --renorm-skip    "${SKIP[@]}" --chunk-mmq
run b-off "${COMMON[@]}" --pcie 1.0 --lora "$A" --lora-exps "${SKIP[@]}"                  # skip forced to 0 (s20 baseline)
run b-on  "${COMMON[@]}" --pcie 1.0 --lora "$A" --lora-exps --renorm-skip "${SKIP[@]}"    # skip 0.15 + rescaled sum
echo "=== gate-renorm done $(date +%T) ===" | tee -a "$D/renorm-gate.summary"
