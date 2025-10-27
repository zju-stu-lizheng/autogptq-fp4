import math
import os
import time
import logging
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer
from .nvfp4_quantizer import NVFP4Quantizer


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


class GPTQ:
    def __init__(self, layer, quant_method="int"):
        self.layer = layer
        self.quant_method = quant_method
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        if self.quant_method == "int":
            self.quantizer = Quantizer()
        elif self.quant_method == "nvfp4":
            self.quantizer = NVFP4Quantizer()

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        inp = inp.to(self.H.device)

        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
        quant_method="int",
        nvfp4_block_size=16,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

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

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                if quant_method == "nvfp4":
                    quantizer.find_params(W[:, i : (i + group_size)], weight=True, block_size=nvfp4_block_size)
                else:
                    quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                scale.append(quantizer.scale)
                if hasattr(quantizer, 'zero'):
                    zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # 获取当前block的权重
            w_block = W1[:, :]  # shape: (rows, blocksize)
            # 设置量化器参数
            if not static_groups:
                if quant_method == "nvfp4":
                    self.quantizer.find_params(w_block, weight=True, block_size=nvfp4_block_size)
                else:
                    self.quantizer.find_params(w_block, weight=True)
                
                scale.append(self.quantizer.scale)
                if hasattr(self.quantizer, 'zero'):
                    zero.append(self.quantizer.zero)
            else:
                if actorder:
                    idx = perm[i1]
                self.quantizer = groups[i1 // group_size]
            
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

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()
        try:
            logger.info(f"duration: {(time.time() - tick)}")
            logger.info(f"avg loss: {torch.sum(Losses).item() / self.nsamples}")
        except Exception as e:
            print(e)
            print(Losses.shape, self.nsamples)
            raise e
        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            if hasattr(self.quantizer, 'zero'):
                zero.append(self.quantizer.zero)
        
        if quant_method == "nvfp4":
            # 对于nvfp4，只返回scale，不返回zero
            scale = torch.cat(scale, dim=1)
            return scale, None, g_idx
        else:
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx

    def pseudo_quantize_to_fp16(self):
        """
        获取伪量化的结果：直接返回已经量化好的权重和scales
        
        Returns:
            tuple: (fp16_weights, per_block_scales, g_idx)
        """
        if self.quant_method != "nvfp4":
            raise ValueError("伪量化只支持nvfp4量化方法")
            
        if not self.quantizer.ready():
            raise ValueError("量化器尚未准备好，请先执行fasterquant")
            
        # 1. 获取已经量化好的权重（已经是float16格式，因为NVFP4Quantizer.quantize返回的是dequantized结果）
        fp16_weights = self.layer.weight.data.clone()
        
        # 2. 获取per-block scales和scale_2
        per_block_scales = self.quantizer.scale.clone()
        scale_2 = self.quantizer.scale_2.clone()
        
        # 3. 将两个缩放因子组合成一个字典或元组
        scales_dict = {
            'per_block_scale': per_block_scales,
            'scale_2': scale_2
        }
        
        # 4. 生成g_idx（基于group_size=16）
        group_size = 16
        g_idx = torch.tensor([i // group_size for i in range(self.columns)], dtype=torch.int32, device=fp16_weights.device)
        
        return fp16_weights, scales_dict, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
