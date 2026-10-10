#!/usr/bin/env bash
# s28: the long-prompt VRAM OOM (Mal's 19K-token request crashed in cudaGraphInstantiate at chunk 12288-16383, 512K
# serve config).  A 50K prompt with the serve flags + GLM_VRAM_TRACE: where does the prefill scratch grow?
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M="/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
name=${1:-oom512}; shift
: > $D/$name.vram
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $D/$name.vram; sleep 0.2; done ) &
smi=$!
cp $HOME/.quetza-data/strata-glm/maya-s-v2.census $D/$name.census
GLM_VRAM_TRACE=1 bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" --backend cuda --experts gpu --arena-gib 72 --arena-skip-resident \
  --chunk-mmq --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-miss 0.15 --skip-file 0.15 \
  --skip-file-prefill 0.15 --renorm-skip --stop 154820,154827,154829 --lora "$A" --lora-exps --pcie auto \
  --census $D/$name.census --dense-requant q5_k --temp 0 -n 16 --ids-file $D/prompts/${IDS:-long50k}.i32 "$@" > $D/$name.log 2>&1
echo "exit $?" >> $D/$name.log
kill $smi
echo "VRAM peak $(sort -n $D/$name.vram | tail -1) MiB" >> $D/$name.log
