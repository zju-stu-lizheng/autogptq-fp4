import torch
from logging import getLogger
logger = getLogger(__name__)

class NVFP4Quantizer:
    """NVFP4量化器，支持block_size=16的per-block量化"""
    
    def __init__(self):
        self.scale = None
        self.scale_2 = None
        self.ready_flag = False
        
        # NVFP4 e2m1格式的边界值和量化值
        self.e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
        self.e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        
    def configure(self, bits=4, perchannel=False, sym=True, mse=False, 
                 norm=2.4, grid=100, maxshrink=0.8, trits=False):
        """配置NVFP4量化器"""
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.trits = trits
        
    def get_weights_scaling_factor_2(self, input_tensor):
        """计算per-tensor的缩放因子2"""
        # 计算全局最大值
        amax = torch.max(torch.abs(input_tensor))
        # NVFP4的缩放因子2 = amax / (6.0 * 448.0)
        return amax.float() / (6.0 * 448.0)
        
    def get_weights_scaling_factor(self, input_tensor, block_size, weights_scaling_factor_2):
        """计算per-block的缩放因子"""
        # 确保输入可以被block_size整除
        assert input_tensor.shape[-1] % block_size == 0, "输入形状不能被block_size整除"
        
        # 计算每个block的最大值
        input_reshaped = input_tensor.view((*tuple(input_tensor.shape[:-1]), -1, block_size))
        per_block_amax = torch.max(torch.abs(input_reshaped), dim=-1)[0].float()
        
        # 计算per-block scale = per_block_amax / (6.0 * weights_scaling_factor_2)
        per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2)
        # 将零值设置为1.0
        per_block_scale[per_block_scale == 0] = 1.0
        
        return per_block_scale
        
    def _cast_fp4(self, weight, device):
        """将权重转换为NVFP4格式"""
        # 将e2m1_bounds和e2m1_values移动到对应设备
        e2m1_bounds = self.e2m1_bounds.to(device)
        e2m1_values = self.e2m1_values.to(device)
        
        # 提取符号位并计算绝对值
        sign_bit = (weight < 0).to(torch.uint8)
        weight_abs = weight.abs()
        
        # 计算序数值
        ord = torch.searchsorted(e2m1_bounds, weight_abs, out_int32=True).to(torch.uint8)
        
        # 检查是否需要舍入到奇数索引的边界值 [0.75, 1.75, 2.5]
        odd_bounds = e2m1_bounds[[1, 3, 5]]  # [0.75, 1.75, 2.5]
        equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
        
        # 组合符号位、序数值和舍入调整
        fp4_values = (sign_bit << 3) + ord + equals_odd_bounds
        
        # 将fp4值转换为实际的浮点数值
        return e2m1_values[fp4_values.long()]
        
    def find_params(self, x, weight=False, block_size=16):
        """为NVFP4格式计算量化参数"""
        dev = x.device
        shape = x.shape
        
        # if weight:
        #     x = x.flatten(1)
        
        # 计算per-tensor缩放因子2
        if weight or self.scale_2 is None:
            self.scale_2 = self.get_weights_scaling_factor_2(x).to(dev)
        
        # 计算per-block缩放因子
        self.scale = self.get_weights_scaling_factor(x, block_size, self.scale_2)
        
        self.ready_flag = True
        
    def quantize(self, x, block_size=16):
        """
        执行NVFP4量化，然后反量化回float16精度
        修复：保证最后reshape不会出错
        """
        if not self.ready_flag:
            return x

        dev = x.device
        original_shape = x.shape

        # 先记录x的2D shape，以便后续恢复
        was_2d_input = (x.dim() == 2)
        batch_size = x.shape[0]
        features = x.shape[1] if was_2d_input else None

        x_flat = x.flatten(1)
        flat_features = x_flat.shape[1]

        # 检查是否需要填充
        pad_size = 0
        if flat_features % block_size != 0:
            pad_size = block_size - (flat_features % block_size)
            x_flat = torch.cat([x_flat, torch.zeros(x_flat.shape[0], pad_size, device=dev, dtype=x.dtype)], dim=1)
            padded = True
        else:
            padded = False

        num_blocks = x_flat.shape[1] // block_size

        # 重塑为block格式
        x_reshaped = x_flat.view(x_flat.shape[0], num_blocks, block_size)  # [B, blocks, block_size]

        # 应用缩放
        scale = self.scale.to(torch.float32)
        scale_2 = self.scale_2.to(torch.float32)
        # scale: [num_blocks] or [1, num_blocks]
        if scale.dim() == 1:
            scale = scale.unsqueeze(0)
        if scale.shape[0] == 1 and x_reshaped.shape[0] > 1:
            scale = scale.expand(x_reshaped.shape[0], -1)

        scale_combined = (scale * scale_2).unsqueeze(-1)  # [B, num_blocks, 1]

        scaled_weight = x_reshaped / scale_combined

        # 转换为NVFP4格式
        fp4_values = self._cast_fp4(scaled_weight, dev)

        # 反量化
        dequantized = fp4_values * scale_combined  # [B, num_blocks, block_size]

        # 展平成2d，去除填充
        dequantized_flat = dequantized.view(x_flat.shape[0], -1)
        if padded:
            dequantized_flat = dequantized_flat[:, :flat_features]

        # 恢复原始形状
        if was_2d_input:
            output = dequantized_flat.view(batch_size, features)
        else:
            output = dequantized_flat.view(original_shape)

        return output

    def quantize_activation(self, x, block_size=16):
        """
        执行NVFP4量化，然后反量化回float16精度，用于激活量化
        支持三维激活(batch, seq, feat)和二维(batch, feat)
        """
        dev = x.device
        original_shape = x.shape

        # 支持3维 (B, S, C) 或2维 (B, C)
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x_reshape = x.contiguous().view(batch_size * seq_len, features)
            reshape_back = True
        else:
            batch_size, features = x.shape
            reshape_back = False
            x_reshape = x

        # 计算scale_2
        scale_2 = self.get_weights_scaling_factor_2(x_reshape).to(dev).to(torch.float32)
        # 计算scale
        scale = self.get_weights_scaling_factor(x_reshape, block_size, scale_2).to(torch.float32)

        # 展平
        x_flat = x_reshape
        flat_features = x_flat.shape[1]

        # 检查是否需要填充
        pad_size = 0
        if flat_features % block_size != 0:
            pad_size = block_size - (flat_features % block_size)
            x_flat = torch.cat(
                [x_flat, torch.zeros(x_flat.shape[0], pad_size, device=dev, dtype=x.dtype)],
                dim=1
            )
            padded = True
        else:
            padded = False

        num_blocks = x_flat.shape[1] // block_size

        # 重塑为block格式 [N, num_blocks, block_size]
        x_blocks = x_flat.view(x_flat.shape[0], num_blocks, block_size)

        # scale形状对齐
        if scale.dim() == 1:
            scale = scale.unsqueeze(0)
        if scale.shape[0] == 1 and x_blocks.shape[0] > 1:
            scale = scale.expand(x_blocks.shape[0], -1)
        scale_combined = (scale * scale_2).unsqueeze(-1)  # [N, num_blocks, 1]

        scaled_weight = x_blocks / scale_combined

        # 转换为NVFP4格式
        fp4_values = self._cast_fp4(scaled_weight, dev)

        # 反量化
        dequantized = fp4_values * scale_combined

        # 展平成2d，去除padding
        dequantized_flat = dequantized.view(x_flat.shape[0], -1)
        if padded:
            dequantized_flat = dequantized_flat[:, :flat_features]

        # 恢复原始形状
        if reshape_back:  # 3维
            output = dequantized_flat.view(batch_size, seq_len, features)
        else:  # 2维
            output = dequantized_flat.view(batch_size, features)

        return output
        
    def ready(self):
        return self.ready_flag

__all__ = ["NVFP4Quantizer"]