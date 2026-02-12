#!/bin/bash

model=your_local_model_path
cp generation_utils.py $model
export HF_ALLOW_CODE_EVAL=1

lengths=(256 512)

port=29510

for l in "${lengths[@]}"
do
    current_port=$((port))
    port=$((port + 1))
    diffusion_steps=$((l))

    accelerate launch --main_process_port ${current_port} eval.py --model dream \
        --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true,decode_method=pbs" \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path "evals_results/humaneval-len${l}_pbs" \
        --log_samples \
        --confirm_run_unsafe_code &> "logs/humaneval-len${l}_pbs.log"
    
    accelerate launch --main_process_port ${current_port} eval.py --model dream \
        --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},temperature=0.0,top_p=0.95,add_bos_token=true,torch_dtype=torch.bfloat16,decode_method=pbs" \
        --tasks mbpp \
        --num_fewshot 3 \
        --batch_size 1 \
        --output_path "evals_results/mbpp-len${l}_ns3_pbs" \
        --log_samples \
        --confirm_run_unsafe_code &> "logs/mbpp-len${l}_ns3_pbs.log"
    
    accelerate launch --main_process_port ${current_port} eval.py --model dream \
        --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},add_bos_token=true,temperature=0.0,top_p=0.95,torch_dtype=torch.bfloat16,decode_method=pbs" \
        --tasks gsm8k_cot \
        --num_fewshot 4 \
        --batch_size 1 \
        --output_path "evals_results/gsm8k-len${l}_ns4_pbs" \
        --log_samples \
        --confirm_run_unsafe_code &> "logs/gsm8k-len${l}_ns4_pbs.log"
    
done


## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
