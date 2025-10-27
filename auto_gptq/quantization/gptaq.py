import math
import os
import time
import logging
from logging import getLogger
import tqdm
import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer
from .nvfp4_quantizer import NVFP4Quantizer
import utils
import quant_utils
import model_utils
# 设置logger写入文件
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 设置日志文件名，可以根据需要更改
log_file = os.path.join(os.path.dirname(__file__), "gptaq.log")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTAQ:

    def __init__(self, layer, quant_method="int"):
        self.layer = layer
        self.quant_method = quant_method
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []
        if self.quant_method == "int":
            self.quantizer = Quantizer()
        elif self.quant_method == "nvfp4":
            self.quantizer = NVFP4Quantizer()

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())

        del self.fp_inp[0]

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, alpha=0.25,
            quant_method="int", nvfp4_block_size=16
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick=time.time()

        if not self.quantizer.ready():
            if quant_method == "nvfp4":
                self.quantizer.find_params(W, weight=True, block_size=nvfp4_block_size)
            else:
                self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # scale it by alpha due to collection of dXXT axnd H
        P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            # for i in range(count):
            #     w = W1[:, i]
            #     d = Hinv1[i, i]

            #     if groupsize != -1:
            #         if not static_groups:
            #             if (i1 + i) % groupsize == 0:
            #                 self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
            #         else:
            #             idx = i1 + i
            #             if actorder:
            #                 idx = perm[idx]
            #             self.quantizer = groups[idx // groupsize]

            #     q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
            #     Q1[:, i] = q
            #     Losses1[:, i] = (w - q) ** 2 / d ** 2

            #     err1 = (w - q) / d
            #     W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
            #     Err1[:, i] = err1
                        # 获取当前block的权重
            w_block = W1[:, :]  # shape: (rows, blocksize)
            # 设置量化器参数
            if not static_groups:
                if quant_method == "nvfp4":
                    self.quantizer.find_params(w_block, weight=True, block_size=nvfp4_block_size)
                else:
                    self.quantizer.find_params(w_block, weight=True)
                
            else:
                if actorder:
                    idx = perm[i1]
                self.quantizer = groups[i1 // groupsize]
            
            # 对整个block进行量化
            if quant_method == "nvfp4":
                q_block = self.quantizer.quantize(w_block, block_size=nvfp4_block_size)
            else:
                q_block = self.quantizer.quantize(w_block)
            
            # 计算block内每列的loss和error，并向后传播
            for local_col in range(count):
                w = w_block[:, local_col]
                q = q_block[:, local_col]
                d = Hinv1[local_col, local_col]
                
                Q1[:, local_col] = q
                Losses1[:, local_col] = (w - q) ** 2 / d**2
                
                # 计算误差并向后传播
                err1 = (w - q) / d
                # W1[:, local_col:] -= err1.unsqueeze(1).matmul(Hinv1[local_col, local_col:].unsqueeze(0))
                Err1[:, local_col] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        torch.cuda.synchronize()
        try:
            logger.info(f"duration: {(time.time() - tick)}")
            logger.info(f"avg loss: {torch.sum(Losses).item() / self.nsamples}")
        except Exception as e:
            print(e)
            print(Losses.shape, self.nsamples)
            raise e

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptaq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTAQ Quantization-----')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module']
    ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)

        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        fp_inputs_cache.clear_hook()
        quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTAQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            handle.remove()

            # copy H and dXXT
            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order,
                    static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTAQ Quantization Done-----\n')

    return quantizers