import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()

def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    import json
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    # 初始化records列表
    records = []

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
    
        for i in range(steps):
            
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # (1) 打印所有mask position的confidence
            mask_positions = torch.where(mask_index[0])[0]
            mask_confidence = confidence[0, mask_positions]
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            selected_positions = []
            selected_confidences = []
            
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                
                # (2) 记录选择了哪个position进行unmask，置信度是多少
                selected_positions.extend(select_index.cpu().float().detach().numpy())
                selected_confidences.extend(confidence[j, select_index].cpu().float().detach().numpy())
                
                # 为每个选择的position添加记录
                for pos, conf in zip(select_index, confidence[j, select_index]):
                    pos_int = pos.item()
                    token = x0[j, pos].item()
                    conf_float = conf.item()
                    
                    records.append({
                        "step": i + 1,
                        "block": num_block + 1,
                        "position": pos_int,
                        "confidence": conf_float,
                        "token_id": token
                    })


            x_before = x.clone()
            x[transfer_index] = x0[transfer_index]
            
            changed_positions = torch.where(x_before[0] != x[0])[0]
    
    return x

def generate_pbs(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, beam_size=2):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        beam_size: Beam size for beam search.
    '''
    import json
    
    # init beam: [(sequence, cumulative_log_prob, block_progress, records)]
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    beam = [(x.clone(), 0.0, 0, [])]  # (sequence, cumulative_log_prob, current_block, records)
    
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for global_step in range(steps):
        new_beam_candidates = []
        
        for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
         
            block_start = prompt.shape[1] + current_block * block_length
            block_end = prompt.shape[1] + (current_block + 1) * block_length
            
            block_mask_index = (seq[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            mask_index = (seq == mask_id)
            if cfg_scale > 0.:
                un_seq = seq.clone()
                un_seq[prompt_index] = mask_id
                seq_ = torch.cat([seq, un_seq], dim=0)
                logits = model(seq_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(seq).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, seq)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            mask_positions = torch.where(mask_index[0])[0]
            mask_confidence = confidence[0, mask_positions]
            
            selected_positions = []
            selected_confidences = []
            transfer_indexs = []
            for j in range(confidence.shape[0]):
                k = beam_size
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    for tmp_select_index in select_index.cpu().numpy().tolist():
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        transfer_index[j, tmp_select_index] = True

                        selected_positions.extend([tmp_select_index])
                        selected_confidences.extend([confidence[j, tmp_select_index].item()])
                        transfer_indexs.extend([transfer_index])
            
            assert len(selected_positions) == len(selected_confidences)
            assert len(selected_positions) == len(transfer_indexs)
            
            if len(selected_positions) > 0:
                for selected_position, selected_confidence, transfer_index in zip(selected_positions, selected_confidences, transfer_indexs):
                    new_seq = seq.clone()
                    new_seq[transfer_index] = x0[transfer_index]
                    
                    token = x0[0, selected_position].item()
                    
                    selected_probs = torch.tensor(selected_confidence, device=seq.device)
                    new_log_prob = cumulative_log_prob + selected_probs.sum().item()
                    
                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1
                    
                    new_records = records.copy()
                    new_records.append({
                        "step": global_step + 1,
                        "position": selected_position,
                        "confidence": selected_confidence,
                        "token_id": token
                    })
                    
                    new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
            else:
                new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                    
        if not new_beam_candidates:
            break
        new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
        
        uniq_new_beam_candidates = []
        seen = set()
        for tensor, float_val, block_progress, records in new_beam_candidates:
            tensor_tuple = tuple(tensor.flatten().cpu().numpy().tolist())
            if tensor_tuple not in seen:
                seen.add(tensor_tuple)
                uniq_new_beam_candidates.append((tensor, float_val, block_progress, records))
        
        beam = uniq_new_beam_candidates[:beam_size]
        
        best_seq, best_score, best_block, best_records = beam[0]

    if beam:
        best_sequence, best_score, _, best_records = beam[0]
    else:
        best_sequence = x
        best_records = []
    
    return best_sequence

def generate_soar(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, max_beam_size=2, log=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        max_beam_size: Maximum beam size for dynamic beam search.
    '''
    confidence_threshold = 0.95
    min_parallel_tokens = 1
    max_parallel_tokens = 5
    
    # init beam: [(sequence, cumulative_log_prob, block_progress, records)]
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    beam = [(x.clone(), 0.0, 0, [])]  # (sequence, cumulative_log_prob, current_block, records)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
        
    for global_step in range(steps):
        has_remaining_masks = False
        for seq, _, _, _ in beam:
            if (seq == mask_id).any():
                has_remaining_masks = True
                break
        
        if not has_remaining_masks:   
            break
        
        beam_sequences = [seq for seq, _, _, _ in beam]
        batch_sequences = torch.cat(beam_sequences, dim=0)  # (beam_size, seq_len)
        
        with torch.no_grad():
            if cfg_scale > 0.:
                unconditional_seqs = []
                for seq in beam_sequences:
                    un_seq = seq.clone()
                    un_seq[prompt_index] = mask_id
                    unconditional_seqs.append(un_seq)
                
                unconditional_batch = torch.cat(unconditional_seqs, dim=0)
                combined_batch = torch.cat([batch_sequences, unconditional_batch], dim=0)
                
                batch_logits = model(combined_batch).logits
                conditional_logits, unconditional_logits = torch.chunk(batch_logits, 2, dim=0)
                batch_logits = unconditional_logits + (cfg_scale + 1) * (conditional_logits - unconditional_logits)
            else:
                batch_logits = model(batch_sequences).logits
        
        logits_with_noise = add_gumbel_noise(batch_logits, temperature=temperature)
        batch_x0 = torch.argmax(logits_with_noise, dim=-1)
        
        if remasking == 'low_confidence':
            p = F.softmax(batch_logits, dim=-1)
            batch_x0_p = torch.gather(p, dim=-1, index=batch_x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            batch_x0_p = torch.rand(batch_x0.shape, device=batch_x0.device)
        else:
            raise NotImplementedError(remasking)
        
        new_beam_candidates = []
        has_multi_unmask_candidate = False
        
        for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
            logits = batch_logits[beam_idx:beam_idx+1]
            x0 = batch_x0[beam_idx:beam_idx+1]
            x0_p = batch_x0_p[beam_idx:beam_idx+1]

            if not (seq == mask_id).any():
                new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                continue
            
            block_start = prompt.shape[1] + current_block * block_length
            block_end = prompt.shape[1] + (current_block + 1) * block_length
            
            mask_index = (seq == mask_id)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf
            
            block_mask_positions = torch.where(mask_index[0, block_start:block_end])[0] + block_start
            block_mask_confidence = confidence[0, block_mask_positions]
            
            high_confidence_mask = block_mask_confidence > confidence_threshold
            high_confidence_indices = torch.where(high_confidence_mask)[0]
            
            if len(high_confidence_indices) >= min_parallel_tokens:
                num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)
                top_probs, top_indices = torch.topk(block_mask_confidence[high_confidence_indices], num_to_unmask)
                selected_indices = high_confidence_indices[top_indices]
                
                new_seq = seq.clone()
                new_log_prob = cumulative_log_prob
                new_records = records.copy()
                
                for idx in range(num_to_unmask):
                    original_idx = selected_indices[idx].item()
                    pos = block_mask_positions[original_idx].item()
                    token = x0[0, pos].item()
                    prob = top_probs[idx].item()
                    
                    new_seq[0, pos] = token
                    new_log_prob += prob
                    
                    new_records.append({
                        "step": global_step + 1,
                        "position": pos,
                        "confidence": prob,
                        "token_id": token
                    })
                
                new_current_block = current_block
                if new_current_block < num_blocks - 1:
                    current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                    if not current_block_mask.any():
                        new_current_block += 1
                
                new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
                has_multi_unmask_candidate = True
                
            else:
                k = min(max_beam_size, len(block_mask_confidence))
                if k == 0:
                    new_current_block = min(current_block + 1, num_blocks - 1)
                    new_beam_candidates.append((seq, cumulative_log_prob, new_current_block, records))
                    continue
                
                top_probs, top_indices = torch.topk(block_mask_confidence, k)
                top_positions = block_mask_positions[top_indices]
                top_tokens = x0[0, top_positions]
                
                for idx in range(k):
                    new_seq = seq.clone()
                    pos = top_positions[idx].item()
                    token = top_tokens[idx].item()
                    prob = top_probs[idx].item()
                    
                    new_seq[0, pos] = token
                    new_log_prob = cumulative_log_prob + prob
                    
                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1
                    
                    new_records = records.copy()
                    new_records.append({
                        "step": global_step + 1,
                        "position": pos,
                        "confidence": prob,
                        "token_id": token
                    })
                    
                    new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
        
        if not new_beam_candidates:
            break
        
        new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
        
        uniq_new_beam_candidates = []
        seen = set()
        for tensor, log_prob, block_progress, records in new_beam_candidates:
            tensor_tuple = tuple(tensor.flatten().cpu().numpy().tolist())
            if tensor_tuple not in seen:
                seen.add(tensor_tuple)
                uniq_new_beam_candidates.append((tensor, log_prob, block_progress, records))
              
        if has_multi_unmask_candidate and uniq_new_beam_candidates:
            best_candidate = uniq_new_beam_candidates[0]
            best_seq, best_log_prob, best_block, best_records = best_candidate
            
            original_mask_count = (beam[0][0] == mask_id).sum().item()
            current_mask_count = (best_seq == mask_id).sum().item()
            masks_unmasked = original_mask_count - current_mask_count
            
            if masks_unmasked >= min_parallel_tokens:
                beam_size = 1
                beam = [best_candidate]
            else:
                beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
                beam = uniq_new_beam_candidates[:beam_size]
        else:
            beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
            beam = uniq_new_beam_candidates[:beam_size]
        
        best_seq, best_score, best_block, best_records = beam[0]
            
    if beam:
        best_sequence, best_score, _, best_records = beam[0]
    else:
        best_sequence = x
        best_records = []
    
    return best_sequence


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
