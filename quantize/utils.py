from collections import OrderedDict
from quantize.int_linear import QuantLinear, PCAQuant, Quant_LoRA_Linear
import torch
import torch.nn as nn
from quantize.int_matmul import QuantMatMul, QuantPCAMatMul

from models.int_llama_layer import QuantLlamaDecoderLayer, QuantLlamaAttention, QuantLlamaMLP

from models.int_llama_layer import  LlamaRMSNorm
from scipy import linalg

from trans_utils import Hadamard_trans
import functools
def get_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params) 

def las_parameters(model, use_shift=True):
    params = []
    template = "learned_scaled_log2"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lab_parameters(model, use_shift=True):
    params = []
    template = "qbias"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lpca_parameters(model, use_shift=True):
    params = []
    template = "PCA_trans"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return params 

def lbl_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('norm') > -1:
            print(n)
            params.append(m)
    return iter(params)  

def lac_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  



def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     



def find_PCA_trans(model, PCA_name):
    for name, module in model.named_modules():
        if isinstance(module, QuantPCAMatMul) or isinstance(module, PCAQuant):
            if PCA_name in name:
                return module.PCA_trans
            
def find_PCA_means(model, PCA_name):
    for name, module in model.named_modules():
        if isinstance(module, PCAQuant):
            if PCA_name in name:
                return module.PCA_means
def fuse_norm(model):
    model.fused=True
    for name, module in model.named_modules():
        if isinstance(module, LlamaRMSNorm):
            if 'input_layernorm' in name:
                for n, m in model.named_modules():
                    if isinstance(m, QuantLinear):
                        if ('q_proj'in n) or ('k_proj'in n) ('q_proj'in n) :
                            m.weight.data = m.weight.data*module.weight

def fuse_R1P(model, P):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            dtype = module.weight.dtype
            P = P.to(device)
            if (('k_proj' in name) or ('q_proj' in name) or ('v_proj' in name))or(('gate_proj' in name) or ('up_proj' in name)) :
                module.weight.data = (module.weight.data.float() @ P).to(dtype)
                    
            if (('o_proj' in name) or ('down_proj' in name)):
                module.weight.data = ((module.weight.data.T.float() @ P).T).to(dtype)
                if module.bias is not None:
                    dtype = module.bias.data.dtype
                    module.bias.data = (module.bias.data.float()@ P).to(dtype)
            
def fuse_R1(model, args,dev):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            dtype = module.weight.dtype
            if (('k_proj' in name) or ('q_proj' in name) or ('v_proj' in name))or(('gate_proj' in name) or ('up_proj' in name)) :
                module.weight.data = Hadamard_trans(module.weight.data).to(dtype)
            if (('o_proj' in name) or ('down_proj' in name)):
                module.weight.data = (Hadamard_trans(module.weight.data.T).T).to(dtype)
                 
def fuse_R3(model, args,dev):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = module.weight.device
            dtype = module.weight.dtype
            if ('o_proj' in name) :
                module.weight.data = Hadamard_trans(module.weight.data, args.head_dim).to(dtype)
            if ('v_proj' in name) :
                module.weight.data = (Hadamard_trans(module.weight.data.T, args.head_dim).T).to(dtype)
                if module.bias is not None:
                    module.bias.data =  Hadamard_trans(module.bias.data, args.head_dim).to(dtype)
                    
 
def fuse_R3_add( model,args, dev):
    hidden_dim = args.hidden_dim
    transdim = args.heads
    R3_add = torch.tensor(linalg.hadamard(transdim))/torch.sqrt(torch.tensor(transdim))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'o_proj'  in name:
                dtype = module.weight.dtype
                R3_add = R3_add.to(dev).to(dtype)
                temp_weight = module.weight.reshape([hidden_dim,args.heads, -1]) 
                temp_weight = temp_weight.transpose(-1,-2)@R3_add
                temp_weight = temp_weight.transpose(-1,-2)
                module.weight.data = temp_weight.reshape([hidden_dim,-1])
    del R3_add

def fuse_R3_slave(model, args, LoRa_dim):
    trans_dim = args.pca_group if args.pca_group <=128 else 128
    hidden_dim = args.hidden_dim
    trans_groups = hidden_dim//trans_dim
    heeddim = args.head_dim
    #print('fuse R3 ',args.R3_quant_params['quant_method'] )
    if args.R3_quant_params['quant_method'] =='Hadamard':
        dev = 'cuda'
        R3 = torch.tensor(linalg.hadamard(trans_dim))/torch.sqrt(torch.tensor(trans_dim))
        R3 = R3.to(dev)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            dtype = module.weight.dtype
            R3 = R3.to(dtype)
                
            if 'o_proj'  in name:
                temp_weight = module.weight.reshape([hidden_dim,trans_groups,trans_dim]) 
                temp_weight = torch.einsum('ohi,ij->ohj',temp_weight,R3) 
                module.weight.data = temp_weight.reshape([hidden_dim,-1])
                
        if isinstance(module, Quant_LoRA_Linear):
            dtype = module.Linear1.weight.dtype
            R3 = R3.to(dtype)
            if ('v_proj' in name) :    
                temp_weight = module.Linear1.weight.data[:,LoRa_dim:].reshape([-1,trans_dim,hidden_dim]) 
                temp_weight = R3@temp_weight 
                module.Linear1.weight.data[:,LoRa_dim:] = temp_weight.reshape([-1,hidden_dim])
                #print(module.Linear2.weight.shape)
                temp_weight = module.Linear1.weight.data[:,:LoRa_dim].reshape([-1,trans_dim,LoRa_dim]) 
                temp_weight = R3@temp_weight 
                module.Linear1.weight.data[:,:LoRa_dim] = temp_weight.reshape([-1,LoRa_dim])
            
        if isinstance(module, QuantPCAMatMul) and 'pv_matmul' in name:
            if args.R3_quant_params['quant_method'] =='PCA':
                del module.PCA_trans
            module.x2_quant_params['quant_method'] = "None"
            if hasattr(module, 'PCA_means'):
                module.PCA_means.data = torch.einsum('hi,hij->hj',module.PCA_means,R3) 
    del R3
    
def fuse_R4(model, args,dev):
    if True:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if ('down_proj' in name):
                    module.weight.data = Hadamard_trans(module.weight.data.detach())
     
    
def PCA_and_quant_temporary(model, args):

    if args.pca_fuse:
        dev = 'cuda'
        R1 = torch.tensor(linalg.hadamard(128))/torch.sqrt(torch.tensor(128))
        R1 = R1.to(dev)
        #M1 = find_PCA_means(model, 'Quant1')
        if args.v_quant_params['quant_method'] =='PCA':
            R3 = find_PCA_trans(model, 'pv')
        else:
            R3 = torch.tensor(linalg.hadamard(128))/torch.sqrt(torch.tensor(128))
            R3 = R3.to(dev)
            R3 = R3.unsqueeze(0)
            R3 = R3.repeat([32,1,1])
        R4 = torch.tensor(linalg.hadamard(128))/torch.sqrt(torch.tensor(128))
        R4 = R4.to(dev)
        #M4 = find_PCA_means(model, 'Quant4')
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                dtype = module.weight.dtype
                R1 = R1.to(dtype)
                R3 = R3.to(dtype)
                R4 = R4.to(dtype)
                module.temp_weight = module.weight
                module.temp_bias = module.bias
                if (('k_proj' in name) or ('q_proj' in name) or ('v_proj' in name)) :
                   
                    
                    temp_weight = module.weight.reshape([-1,32,128]) 
                    module.temp_weight = torch.einsum('ohi,ij->ohj',temp_weight,R1) 
                    
                    if ('v_proj' in name) :
                        
                        
                        if module.weight.shape[0] != module.weight.shape[1]:
                            R3_ = R3.reshape([8,4,128,128])[:,0,:,:]
                        else:
                            R3_ = R3
                        temp_weight = module.temp_weight.reshape([-1,128,4096]) 
                        module.temp_weight = torch.einsum('hoi,hoj->hji',temp_weight,R3_)
                        
                    module.temp_weight = module.weight_quantizer(module.temp_weight.reshape([-1,4096]))
                    
                    if ('v_proj' in name) :
                        temp_weight = module.temp_weight.reshape([-1,128,4096]) 
                        module.temp_weight = torch.einsum('hoi,hjo->hji',temp_weight,R3_)
                        
                    temp_weight = module.temp_weight.reshape([-1,32,128]) 
                    module.temp_weight = torch.einsum('ohi,ji->ohj',temp_weight,R1) 
                    module.temp_weight = module.temp_weight.reshape([-1,4096])
                    
                    
                if 'o_proj'  in name:
                    
                    temp_weight = module.weight.reshape([4096,32,128]) 
                    module.temp_weight = torch.einsum('ohi,hij->ohj',temp_weight,R3) 
                        
                    
                    #module.temp_bias = -M1.reshape([-1])
                    #module.temp_bias = torch.einsum('hi,hij->hj',module.temp_bias,R1) 
                    temp_weight = module.temp_weight.reshape([32,128,4096]) 
                    module.temp_weight = torch.einsum('hoi,oj->hji',temp_weight,R1) 
                    
                    module.temp_weight = module.weight_quantizer(module.temp_weight.reshape([4096,-1]))
                    
                    temp_weight = module.temp_weight.reshape([4096,32,128]) 
                    module.temp_weight = torch.einsum('ohi,hji->ohj',temp_weight,R3) 
                    
                    temp_weight = module.temp_weight.reshape([32,128,4096]) 
                    module.temp_weight = torch.einsum('hoi,jo->hji',temp_weight,R1) 
                    
                    module.temp_weight = module.temp_weight.reshape([4096,-1])
                    
                    
                    
                if  (('gate_proj' in name) or ('up_proj' in name)):
                    #module.temp_bias = M1.reshape([-1])@module.weight.data.t()
                    
                    temp_weight = module.weight.reshape([-1,32,128]) 
                    module.temp_weight = torch.einsum('ohi,ij->ohj',temp_weight,R1) 
                    module.temp_weight = module.weight_quantizer(module.temp_weight.reshape([-1,4096]))
                    temp_weight = module.temp_weight.reshape([-1,32,128]) 
                    module.temp_weight = torch.einsum('ohi,ji->ohj',temp_weight,R1) 
                    
                    
                    module.temp_weight = module.temp_weight.reshape([-1,4096])
                    
                    
                if (args.fuse4 )and('down_proj' in name) :
                    #module.temp_bias = M4.reshape([-1])@module.weight.data.t()
                    
                    temp_weight = module.weight.reshape([4096,-1,128]) 
                    module.temp_weight = torch.einsum('ohi,ij->ohj',temp_weight,R4)
                    temp_weight = module.temp_weight.reshape([32,128,-1]) 
                    module.temp_weight = torch.einsum('hoi,oj->hji',temp_weight,R1) 
                    module.temp_weight = module.weight_quantizer(module.temp_weight.reshape([4096,-1]))
                    temp_weight = module.temp_weight.reshape([4096,-1,128]) 
                    module.temp_weight = torch.einsum('ohi,ji->ohj',temp_weight,R4) 
                    temp_weight = module.temp_weight.reshape([32,128,-1]) 
                    module.temp_weight = torch.einsum('hoi,jo->hji',temp_weight,R1) 
                    module.temp_weight = module.temp_weight.reshape([4096,-1])
                    
                module.use_temporary_parameter=True
                #if 'o_' in name:
                #print(name,module.weight_quantizer.upbound_factor.reshape([-1])[0:4])
        del R1
        del R3
        del R4            
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight_quantizer(module.weight)
                module.temp_bias = module.bias
       
    
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias



def PCA_and_quant_implace(model, args):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.temp_weight
            module.use_temporary_parameter=False
    
    
def set_quant_state(self, act_quant: bool = False, quant_mode=0):

    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (PCAQuant, QuantPCAMatMul)):
            m.set_quant_state(act_quant)
        if isinstance(m, (QuantLlamaAttention, QuantLlamaMLP)):
            m.use_act_quant = act_quant

def set_KVquant_state(self, act_quant: bool = False, quant_mode=0):

    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (PCAQuant, QuantPCAMatMul)):
            m.set_quant_state(act_quant)


def set_fuse_state(self, args, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for name, m in self.named_modules():
        if 'Quant1' in name and isinstance(m, PCAQuant):
            if args.fuse1:
                m.set_quant_state(weight_quant, True)
            else:
                m.set_quant_state(weight_quant, False)
                
        if isinstance(m, QuantLlamaAttention):
            m.use_act_quant=True
                
        if isinstance(m, QuantLlamaDecoderLayer):

            m.use_act_quant = True
                
        if 'Quant4' in name and isinstance(m, PCAQuant):
            if args.fuse4:
                m.set_quant_state(weight_quant, True)
            else:
                m.set_quant_state(weight_quant, False)

def get_act_means(model, dataloader, num_samples, bsz, keys, attention_mask,position_ids):
    model.eval()
    device = next(model.parameters()).device
    act_disturb = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach().cpu()
        if name in act_disturb:
            act_disturb[name] = torch.cat((act_disturb[name], tensor.to(torch.float32).to('cpu')), dim=0)
        else:
            act_disturb[name] = tensor.to(torch.float32).to('cpu')

    def stat_input_hook(m, x, y, name):
        # 捕获输入
        
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        for key in keys:
            if isinstance(m, nn.Linear) and ((key in name)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=key))
                )

    for i in range(num_samples//bsz):
        model(dataloader[i*bsz:(i+1)*bsz].to(device), attention_mask=attention_mask,position_ids=position_ids)


    for h in hooks:
        h.remove()

    return act_disturb
