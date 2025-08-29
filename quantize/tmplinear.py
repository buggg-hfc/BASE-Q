import torch.nn as nn
import torch
from trans_utils import Hadamard_trans
from quantize.int_linear import *
class TmpLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear
    ):
        super(TmpLinear, self).__init__()
        self.ori_weight = org_module.weight.data.clone()
        self.weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.register_buffer('ori_bias',org_module.bias)
        else:
            self.ori_bias = None
            self.bias = None
        
        self.fwd_func = F.linear
        self.quantizer = Quantizer()
        self.quantizer.configure(
                4, perchannel=True, sym=False, mse=True
        )
        self.input_trans = False
        self.output_trans = False
        self.dim = -1
        self.rotation = None
    def round_ste(self, x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x

    def quantize(self, x, scale, zero, maxq):
        #print(x.shape,scale.shape,zero.shape,maxq.shape)
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(self.round_ste(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
    
    def find_params(self, a, a2=1.):
        #print(a.shape,self.ori_weight.shape)
        self.weight = self.ori_weight * a
        self.weight = self.weight.detach()
        if self.input_trans:
            if self.rotation is not None:
                self.weight = self.rotation(self.weight)
            self.weight = Hadamard_trans(self.weight,self.dim)
        if self.output_trans:
            if self.rotation is not None:
                self.weight = self.rotation(self.weight.T).T
            self.weight = Hadamard_trans(self.weight.T,self.dim).T
        self.weight = self.weight * a2
        self.quantizer.find_params(self.weight.detach(), weight=True)

    def quant_tmpweight(self, a, a2=1.):
        self.weight = self.ori_weight * a
        if self.ori_bias is not None:
            self.bias = self.ori_bias.clone()
        if self.input_trans:
            if self.rotation is not None:
                self.weight = self.rotation(self.weight)
            self.weight = Hadamard_trans(self.weight,self.dim)
        if self.output_trans:
            if self.rotation is not None:
                self.weight = self.rotation(self.weight.T).T
            self.weight = Hadamard_trans(self.weight.T,self.dim).T
            if self.ori_bias is not None:
                if self.rotation is not None:
                    self.bias = self.rotation(self.bias)
                self.bias = Hadamard_trans(self.bias,self.dim)
                self.bias = self.bias * a2
        self.weight = self.weight * a2
        self.weight = self.quantize(self.weight, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)

    def quant_tmpweight_rtn(self,a):
        # 使用模型
        self.weight = self.ori_weight * a
        self.weight = self.quantize(self.weight, self.quantizer.scale*2*F.sigmoid(self.quantizer.alpha), self.quantizer.zero, self.quantizer.maxq)
        #print(self.tmp_weight[0,0:16])
    def forward(self, input: torch.Tensor):
        return self.fwd_func(input, self.weight, self.bias)


def get_Tmp_quantizer_params(model):
    scale_dict = {}
    zero_dict = {}
    for n,m in model.named_modules():
        if type(m) == TmpLinear:
            scale_dict[n] = (m.quantizer.scale*2*F.sigmoid(m.quantizer.alpha)).detach()
            zero_dict[n] = m.quantizer.zero.detach().clone()
    return scale_dict, zero_dict


def replace_TmpLinear_with_Linear(model):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
                module = torch.nn.Linear(in_features = m.ori_weight.data.shape[1], out_features =  m.ori_weight.data.shape[0], bias = m.ori_bias is not None)
                module.weight.data = m.ori_weight.data
                old_m = getattr(model, n)
                if module.bias is not None:
                    module.bias.data = m.ori_bias.data
                setattr(model, n, module)
                print("Succesee replace module to Linear:",n)
                del old_m
        else:
            replace_TmpLinear_with_Linear(m)

def replace_o_proj_with_Linear(model):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            if 'o_proj' in n :
                module = torch.nn.Linear(in_features = m.ori_weight.data.shape[1], out_features =  m.ori_weight.data.shape[0], bias = m.ori_bias is not None)
                module.weight.data = m.ori_weight.data
                old_m = getattr(model, n)
                if module.bias is not None:
                    module.bias.data = m.ori_bias.data
                setattr(model, n, module)
                print("Succesee replace module to Linear:",n)
                del old_m
        else:
            replace_o_proj_with_Linear(m)

def replace_v_proj_with_Linear(model):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            if 'v_proj' in n :
                module = torch.nn.Linear(in_features = m.ori_weight.data.shape[1], out_features =  m.ori_weight.data.shape[0], bias = m.ori_bias is not None)
                module.weight.data = m.ori_weight.data
                old_m = getattr(model, n)
                if module.bias is not None:
                    module.bias.data = m.ori_bias.data
                setattr(model, n, module)
                print("Succesee replace module to Linear:",n)
                del old_m
        else:
            replace_v_proj_with_Linear(m)

def replace_down_proj_with_Linear(model):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            if 'down_proj' in n :
                module = torch.nn.Linear(in_features = m.ori_weight.data.shape[1], out_features =  m.ori_weight.data.shape[0], bias = m.ori_bias is not None)
                module.weight.data = m.ori_weight.data
                old_m = getattr(model, n)
                if module.bias is not None:
                    module.bias.data = m.ori_bias.data
                setattr(model, n, module)
                print("Succesee replace module to Linear:",n)
                del old_m
        else:
            replace_down_proj_with_Linear(m)

def replace_linear_with_TmpLinear(model):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
            
                setattr(model, n, TmpLinear(m))
             
        else:
            replace_linear_with_TmpLinear(m)

def replace_o_proj_with_TmpLinear(model):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if  'o_proj' in n:
                setattr(model, n, TmpLinear(m))
        else:
            replace_o_proj_with_TmpLinear(m)
def replace_down_proj_with_TmpLinear(model):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if  'down_proj' in n:
                setattr(model, n, TmpLinear(m))
        else:
            replace_down_proj_with_TmpLinear(m)

            
def replace_v_proj_with_TmpLinear(model):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if  'v_proj' in n:
                setattr(model, n, TmpLinear(m))
        else:
            replace_v_proj_with_TmpLinear(m)

def replace_linear_with_Quantlinear(model):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            setattr(model, n, QuantLinear(m))
        else:
            replace_linear_with_Quantlinear(m)
            
def replace_linear_with_WQuantlinear(model, wbits):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            setattr(model, n, WQuantLinear(m, wbits))
        else:
            replace_linear_with_WQuantlinear(m, wbits)

def replace_WQuantlinear_with_linear(model):
    for n,m in model.named_children():
        if isinstance(m, WQuantLinear):
            module = torch.nn.Linear(in_features = m.weight.data.shape[1], out_features =  m.weight.data.shape[0], bias = m.bias)
            module.weight.data = m.weight.data
            if module.bias is not None:
                module.bias.data = m.bias.data
            setattr(model, n, module)
        else:
            replace_WQuantlinear_with_linear(m)