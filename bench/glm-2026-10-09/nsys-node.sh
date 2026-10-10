#!/usr/bin/env bash
# s28: decode kernel time WITH CUDA-graph nodes expanded (s27 errata: the old capture left the attention/finish/predict
# graph nodes out of the kernel table).  512K serve flags, chat_code6k prompt, 64 decode tokens.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M="/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
O=$D/node-decode
cp $HOME/.quetza-data/strata-glm/maya-s-v2.census $D/node.census
GLM_PROFILE_DECODE=1 flock "$HOME/.quetza-data/conductor/ds4-gpu.lock" systemd-run --user --scope -q -p MemoryMax=84G -p MemorySwapMax=0 \
  /usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --sample=none \
  --cpuctxsw=none --cuda-graph-trace=node -o $O -f true \
  "$B" -m "$M" --backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --chunk-mmq \
  --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-miss 0.15 --skip-file 0.15 \
  --skip-file-prefill 0.15 --renorm-skip --stop 154820,154827,154829 --lora "$A" --lora-exps --pcie auto \
  --census $D/node.census --dense-requant q5_k --temp 0 -n 64 --ids-file $D/prompts/chat_code6k.i32 \
  --ctx 524288 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 > $O.log 2>&1
echo "nsys exit $?" >> $O.log
/usr/local/cuda/bin/nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace -f csv -o $O $O.nsys-rep > /dev/null 2>&1
ls $O*.csv >> $O.log
