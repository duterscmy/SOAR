
#!/bin/bash

lengths=(256 512)
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
script=eval_llada.soar.py

for length in "${lengths[@]}"; do
    python $script --tasks humaneval --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=$length,steps=$length,block_length=$length,low_cpu_mem_usage=True,device_map='auto' \
        &> logs/humaneval.len${length}.soar.log

    python $script --tasks mbpp --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=$length,steps=$length,block_length=$length,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16 \
        &> logs/mbpp.len${length}.soar.log

    python $script --tasks gsm8k --model llada_dist \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=$length,steps=$length,block_length=$length,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16 \
        &> logs/gsm8k.len${length}.soar.log
    
done