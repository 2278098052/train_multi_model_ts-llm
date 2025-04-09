import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
def model(input_tensor: torch.Tensor) -> torch.Tensor:

    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)  
    prob_dist = torch.rand(batch_size, seq_len, 151246)    # Random values in [0, 1)    # Normalize to sum to 1 per row
    return prob_dist
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
def generate(tensor):
    tokenizer = AutoTokenizer.from_pretrained('LLM/Qwen2.5-0.5B-Instruct')
    max_seq_len = tensor.size(1)+100
    bsz = 8
    mask = (tensor == 151643)
    first_pad_indices = mask.int().argmax(dim=1)
    no_mask = ~mask.any(dim=1)
    first_pad_indices[no_mask] = tensor.size(1)
    min_prompt_len = first_pad_indices.min()
    max_prompt_len = first_pad_indices.max()

    total_len = max_seq_len
    pad_id = 151643
    tokens = tensor
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz)
    input_text_mask = tokens != pad_id

    temperature = 0.8
    top_p = 0.6
    for cur_pos in range(min_prompt_len, total_len):
        logits = model(tokens[:, prev_pos:cur_pos])#(8, 151246)

        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)#(8,1)


        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated

        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == 151645 #<|im_end|>
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break
    print(tokenizer.batch_decode(tokens))    

def generate_tensor(batch_size, seq_len, pad_value=151643):
    # 1. 生成随机截断点（每个序列一个）
    trunc_points = torch.randint(1, seq_len + 1, (batch_size,))
    
    # 2. 生成随机整数张量（范围 0-100）
    random_values = torch.randint(0, 15000, (batch_size, seq_len))
    
    # 3. 生成掩码（mask），标记哪些位置需要替换为 pad_value
    mask = torch.arange(seq_len).expand(batch_size, -1) >= trunc_points.unsqueeze(1)
    
    # 4. 应用掩码，替换为 pad_value
    tensor = torch.where(mask, pad_value, random_values)
    
    return tensor
x = generate_tensor(8,100,151643)
y = generate(x)

print(y.shape)  # Should be (8, 151246)
print(y)