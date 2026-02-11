from numpy import concat, require
import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights, apply_rope_vectorized


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 16            # Number of transformer layers

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path,
            local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        self.kv_cache = {}
        self.kv_cache_lengths = None
    
    def run(self, input_ids, prefill = True):
        ########################################
        # Complete this function
        ########################################
        N = len(input_ids)
        lengths = torch.tensor([len(ids) for ids in input_ids], device='cuda')
        CT = lengths.max()
        if prefill:
            input_tensor = torch.zeros(N, CT, dtype=input_ids[0].dtype, device='cuda')
            for i in range(N):
                input_tensor[i, :lengths[i]] = input_ids[i]
            self.kv_cache_lengths = torch.zeros(N, dtype=torch.int64, device='cuda')
            self.kv_cache = {}
        else:
            assert self.kv_cache_lengths is not None
            input_tensor = torch.tensor(input_ids).reshape(N, CT)
        hidden_state = self.weights["embedding"][input_tensor] # (N, CT, D)
        D = hidden_state.shape[-1]
        D2 = self.weights["self_attn_k_proj_weight"][0].shape[0]
        PT = self.kv_cache_lengths.max()

        lengths_mask = torch.arange(CT, device='cuda').repeat(N, 1) >= lengths.unsqueeze(1) # (N, CT)
        hidden_state[lengths_mask] = 0

        for current_layer in range(self.layers):
            # --- Self-Attention Block ---
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5) # (N, CT, 1)
            normalized_x = hidden_state / rms # (N, CT, D)
            x = normalized_x.to(torch.float16) * self.weights["layernormAttn_weight"][current_layer] # (N, CT, D)
            
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t()) # (N, CT, D2)
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t()) # (N, CT, D2)
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t()) # (N, CT, D2)
            
            # Apply RoPE to query and key using the helper function
            apply_rope_vectorized(q, output=q, head_dim=self.head_dim, offset=self.kv_cache_lengths)
            apply_rope_vectorized(k, output=k, head_dim=self.head_dim, offset=self.kv_cache_lengths)
            
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            
            sub_q = q.view(N, CT, self.num_qo_heads, self.head_dim).permute(1, 0, 2, 3) # (CT, N, self.num_qo_heads, self.head_dim)
            sub_k = k.view(N, CT, self.num_kv_heads, self.head_dim).permute(1, 0, 2, 3) # (CT, N, self.num_kv_heads, self.head_dim)
            sub_v = v.view(N, CT, self.num_kv_heads, self.head_dim).permute(1, 0, 2, 3) # (CT, N, self.num_kv_heads, self.head_dim)

            # use kv cache to get full kv
            if not prefill:
                sub_k_new = torch.cat([self.kv_cache[current_layer][0], torch.zeros(CT, N, self.num_kv_heads, self.head_dim, dtype=sub_k.dtype, device=sub_k.device)], dim=0) # (offset + CT, N, self.num_kv_heads, self.head_dim)
                sub_v_new = torch.cat([self.kv_cache[current_layer][1], torch.zeros(CT, N, self.num_kv_heads, self.head_dim, dtype=sub_v.dtype, device=sub_v.device)], dim=0) # (offset + CT, N, self.num_kv_heads, self.head_dim)
                sub_k_new[self.kv_cache_lengths, torch.arange(N)] = sub_k.squeeze(0)
                sub_v_new[self.kv_cache_lengths, torch.arange(N)] = sub_v.squeeze(0)
                sub_k = sub_k_new
                sub_v = sub_v_new
            
            self.kv_cache[current_layer] = (sub_k, sub_v)
            
            n_q = sub_q.shape[0] # CT
            n_k = sub_k.shape[0] # PT + CT
            assert n_q == CT
            assert n_k == PT + CT

            sub_k = sub_k.repeat_interleave(group_size, dim=-2) # (PT + CT, N, self.num_qo_heads, self.head_dim)
            sub_v = sub_v.repeat_interleave(group_size, dim=-2) # (PT + CT, N, self.num_qo_heads, self.head_dim)
            
            sub_q_t = sub_q.permute(1, 2, 0, 3) # (N, self.num_qo_heads, CT, self.head_dim)
            sub_k_t = sub_k.permute(1, 2, 0, 3) # (N, self.num_qo_heads, PT + CT, self.head_dim)

            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale # (N, self.num_qo_heads, CT, PT + CT)

            if prefill:
                mask = torch.ones((n_q, n_k), dtype=torch.bool, device='cuda').triu(diagonal=1) # (CT, PT + CT)
                scores = scores.masked_fill(mask, float('-inf')) # valid because if prefill, offset = 0.
                # we don't mask out the scores because we mask it out at attn_output.
            else:
                attn_weights_mask = torch.arange(PT + CT, device='cuda').repeat(CT, 1) >= (self.kv_cache_lengths + lengths).unsqueeze(1) # (N, CT)
                scores = scores.masked_fill(attn_weights_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1) # (N, self.num_qo_heads, CT, PT + CT)

            v_t = sub_v.permute(1, 2, 0, 3) # (N, self.num_qo_heads, PT + CT, self.head_dim)
            attn_output = torch.matmul(attn_weights, v_t) # (N, self.num_qo_heads, CT, self.head_dim)
            attn_output = attn_output.masked_fill(lengths_mask.unsqueeze(1).unsqueeze(3), 0)

            attn_output = attn_output.permute(0, 2, 1, 3) # (N, CT, self.num_qo_heads, self.head_dim)
            attn_output = attn_output.reshape(N, n_q, self.num_qo_heads * self.head_dim) # (N, CT, self.num_qo_heads * self.head_dim) = (N, CT, D)

            attn_output = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t()) + hidden_state # (N, CT, D)

            # --- Feed-Forward Network (FFN) Block ---
            rms = torch.sqrt(torch.mean(attn_output ** 2, dim=-1, keepdim=True) + 1e-5) # (N, CT, 1)
            normalized_x = attn_output / rms # (N, CT, D)
            layernormFFN_output = normalized_x.to(torch.float16) * self.weights["layernormFFN_weight"][current_layer]
            
            up_proj_output = layernormFFN_output.matmul(self.weights["up_proj_weight"][current_layer].t())
            gate_proj_output = layernormFFN_output.matmul(self.weights["gate_proj_weight"][current_layer].t())
            
            activation_output = up_proj_output * torch.nn.functional.silu(gate_proj_output)
            hidden_state = activation_output.matmul(self.weights["down_proj_weight"][current_layer].t()) + attn_output

        # --- Final Layer Normalization and Output Projection ---
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        
        self.kv_cache_lengths += lengths
        sample_output = torch.argmax(logits, dim=-1)
        return sample_output[torch.arange(N), lengths - 1].clone().to(device='cpu')
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)
            
        output_ids_list = input_ids_list  
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)

        for round in range(rounds - 1):
            print(f"Round {round}")
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)
            
            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat((output_ids_list[i], new_token[i:i+1]), dim=0)
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        return output_text_list

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "The University of Washington is located in"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)