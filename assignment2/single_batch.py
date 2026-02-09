import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


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
    
    def run(self, input_ids, prefill = True):
        ########################################
        # Complete this function
        ########################################
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        hidden_state = self.weights["embedding"][input_tensor]

        if prefill:
            # clear kv cache
            self.kv_cache = {}
            offset = 0
        else:
            # get offset from kv cache for rope
            offset = self.kv_cache[0][0].shape[0]

        hidden_state = self.weights["embedding"][input_tensor]

        for current_layer in range(self.layers):        
            # --- Attention Block ---
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x.to(torch.float16) * self.weights["layernormAttn_weight"][current_layer]
            
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            
            
            # Apply RoPE
            apply_rope(q, output=q, head_dim=self.head_dim, offset=offset)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=offset)

            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            
            # Reshape for heads
            sub_q = q.view(-1, self.num_qo_heads, self.head_dim) # (curr_seq_len, num_qo_heads, head_dim)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim) # (curr_seq_len, num_kv_heads, head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim) # (curr_seq_len, num_kv_heads, head_dim)

            # use kv cache to get full kv
            if not prefill:
                sub_k = torch.cat([self.kv_cache[current_layer][0], sub_k], dim=0)
                sub_v = torch.cat([self.kv_cache[current_layer][1], sub_v], dim=0)
            
            self.kv_cache[current_layer] = (sub_k, sub_v)
            
            n_q = sub_q.shape[0]
            n_k = sub_k.shape[0]    

            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)

            sub_q_t = sub_q.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            sub_k_t = sub_k.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)

            score = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale # (num_qo_heads, seq_len, seq_len)

            if prefill:
                mask = torch.ones((n_q, n_k), dtype=torch.bool, device='cuda').triu(diagonal=1)
                score = score.masked_fill(mask, float('-inf'))
            # For non-prefill, we only care about the last token's score, so no mask needed
            
            attn_weights = torch.softmax(score, dim=-1)

            v_t = sub_v.permute(1, 0, 2) # (num_qo_heads, seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, v_t) # (num_qo_heads, seq_len, head_dim)

            attn_output = attn_output.permute(1, 0, 2) # (seq_len, num_qo_heads, head_dim)
            attn_output = attn_output.reshape(n_q, self.num_qo_heads * self.head_dim) # (seq_len, num_qo_heads * head_dim)

            attn_output = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t()) + hidden_state

            # --- Feed-Forward Network (FFN) Block ---
            rms = torch.sqrt(torch.mean(attn_output ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = attn_output / rms
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
        
        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()
        
    
    def generate(self, input_string, rounds=20):
        input_ids = self.tokenizer.encode(input_string)

        # print("Token IDs:", input_ids)
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            # print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=20)
    print("Generated Text:", output_text)