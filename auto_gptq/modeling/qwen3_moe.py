from ._base import BaseGPTQForCausalLM


class Qwen3MOEGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Qwen3MoEDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm", ]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ## "Gate and up proj"
        ### 128 experts
        [ "mlp.experts.{}.gate_proj".format(i) for i in range(128) ] + [ "mlp.experts.{}.up_proj".format(i) for i in range(128) ],
        [ "mlp.experts.{}.down_proj".format(i) for i in range(128) ],
    ]

__all__ = ["Qwen3MOEGPTQForCausalLM"]
