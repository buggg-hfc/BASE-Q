# This file is modified from https://github.com/OpenGVLab/OmniQuant/quantize/omniquant.py

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.tmplinear import *
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import  lac_parameters, lab_parameters, get_parameters,\
                            clear_temp_variable,set_quant_state,\
                            fuse_R3, fuse_R4, fuse_R1P, get_act_means


import tqdm

import functools
from scipy import linalg

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import os


from scipy import linalg

# GPTQ
from gptq.gptq import *
from gptq.modelutils import *
from gptq.quant import *

from trans_utils import Hadamard_trans, ORTransMatrix, pca_cov, PCA_rotation
    
 
import torch.nn.functional as F


def GPTQ_process(qlayer, args, quant_inps, attention_mask, position_ids, logger, dev):
    with torch.no_grad():
        full = find_layers(qlayer)
        #print(full)
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
    
        for names in sequential:
            subset = {n: full[n] for n in names} 

            gptq = {}
            
            for name in subset:  
                
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.w_sym, mse= args.mse
                )
                if args.scale_keep:
                    for key in scale_dict:
                        if name in key:
                            print('keep scale in',name)
                            gptq[name].quantizer.scale = scale_dict[key].clone()
                            gptq[name].quantizer.zero = zero_dict[key].clone()
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name))) 

            for j in range(128):
                qlayer(quant_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids)[0] 
            for h in handles:
                h.remove() 

            for name in subset:
                print('Quantizing ...', name)
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups,
                    LoRa= 'Linear1' in name, LoRa_dim=64, Use_RTN = args.Use_RTN
                )
                gptq[name].free()
        del gptq 

def baseq(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if args.info:
        print(args)
        print(model)
        print(type(model))

    if "llama" in args.net.lower() or "qwen" in args.net.lower(): 
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for qwen2.5, llama-2, Llama-3/3.1/3.2 now")
    

    layers[0] = layers[0].to(dev)

    if args.device_map == "cpu":
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    else:
        class RotaryWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hidden_states, position_ids):
                if self.module.inv_freq.device != hidden_states.device:
                    self.module.inv_freq = self.module.inv_freq.to(hidden_states.device)
                return self.module(hidden_states, position_ids)

        model.model.rotary_emb = RotaryWrapper(model.model.rotary_emb)
  
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = args.dtype
        traincast = torch.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            
            raise ValueError

    
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    # move embedding layer and first layer to cpu
    
    layers[0] = layers[0].module 
    layers[0] = layers[0].cpu() 
    
    

    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for qwen2.5, llama-2 and Llama-3/3.1/3.2 now")
    torch.cuda.empty_cache()
    
    # same input of first layer for fp model and quant model
    inps = inps.to('cpu')
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_outs = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None


        
        
        
    #### Fuse parameters of RMSNorm and Rotation, abtain new model arch 
    logger.info(f"=== Start fuse nrom layers ===")
    for i in (range(len(layers))):
        layer = layers[i].to(dev)
        for n,m in layer.named_modules():
            if 'input_layernorm' in n:
                for name, module in  layer.named_modules():
                    if (isinstance(module, nn.Linear)) and( ('q_proj' in name) or ('k_proj' in name) or ('v_proj' in name)):
                        module.weight.data = module.weight.data * m.weight
                        
                m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)
                
            if 'post_attention_layernorm' in n:
                for name, module in  layer.named_modules():
                    if (isinstance(module, nn.Linear)) and( ('up_proj' in name) or ('gate_proj' in name)):
                        module.weight.data = module.weight.data * m.weight
                m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)

        if "llama" in args.net.lower() or "qwen" in args.net.lower():  
            qlayer = DecoderLayer(lm.model.config, layer, args) 
        qlayer = qlayer.to(dev).to(dtype)

        layers[i] = qlayer.to("cpu")
        del qlayer
    
    
    logger.info(f"=== Start fuse rotation layers ===")
    if args.scale_load_dir:
        scale_dict = torch.load(args.scale_load_dir, map_location=dev)

    if args.Rres_init:
        if args.Rres_init == 'PCA':
            print("Doing PCA Rres Init...")
            weight_crossvar = {}
            for name, module in layers.named_modules():
                if type(module) == nn.Linear and name.split('.')[-1] in ['q_proj','k_proj','v_proj','up_proj','gate_proj']:
                    w = module.weight.data.clone().double()    
                    if 'res' in weight_crossvar:
                        weight_crossvar['res'] += (w.T@w).to('cpu')
                    else:
                        weight_crossvar['res'] = (w.T@w).to('cpu')
            H = weight_crossvar['res']
            L, P = pca_cov(H)
            P = Hadamard_trans(P.float())
        elif args.Rres_init == 'Spinquant':
            P = torch.load(args.Rres_init_path)['R1']
        else:
            P = torch.eye(args.hidden_dim)
            P = Hadamard_trans(P.float())
    
        
    if not args.disable_rotation:
        for i in (range(len(layers))):
            qlayer = layers[i].to(dev)
            if args.scale_load_dir:
                if args.scale_imask[0]=='1':
                    scale_o = scale_dict['model.layers.'+str(i)+'.self_attn.o_proj'] 
                    dtype = qlayer.self_attn.v_proj.weight.data.dtype
                    scale = scale_o
                    if qlayer.self_attn.v_proj.weight.data.shape[0]== qlayer.self_attn.v_proj.weight.data.shape[1]:
                        qlayer.self_attn.v_proj.weight.data = (qlayer.self_attn.v_proj.weight.data.double()/scale.unsqueeze(1).double()).to(dtype)
                        qlayer.self_attn.o_proj.weight.data = (qlayer.self_attn.o_proj.weight.data.double()*scale.double()).to(dtype)
                    else:
                        if args.old_scale == False:
                            qlayer.self_attn.v_proj.weight.data = (qlayer.self_attn.v_proj.weight.data.double()/scale.unsqueeze(1).double()).to(dtype)
                            scale = scale.reshape(-1,128).repeat_interleave(repeats=args.kv_group,dim=0).reshape(-1)
                            qlayer.self_attn.o_proj.weight.data = (qlayer.self_attn.o_proj.weight.data.double()*scale.double()).to(dtype)
                        else:
                            o_scale = scale.unsqueeze(1)
                            o_scale = o_scale.repeat(1,128)
                            o_scale = o_scale.flatten()
                            qlayer.self_attn.o_scale.data = (1/o_scale.double()).to(dtype)
                            tmp = qlayer.self_attn.o_proj.weight.data 
                            tmp = tmp.reshape(8192,64,128)
                            tmp = tmp*scale.unsqueeze(1)
                            qlayer.self_attn.o_proj.weight.data = tmp.reshape(8192,8192).to(dtype)
                if args.scale_imask[1]=='1':
                    scale_down = scale_dict['model.layers.'+str(i)+'.mlp.down_proj'] 
                    dtype = qlayer.mlp.up_proj.weight.data.dtype
                    scale = scale_down.to(qlayer.mlp.up_proj.weight.data)
                    qlayer.mlp.up_proj.weight.data = (qlayer.mlp.up_proj.weight.data.double()/scale.unsqueeze(1).double()).to(dtype)
                    qlayer.mlp.down_proj.weight.data = (qlayer.mlp.down_proj.weight.data.double()*scale.double()).to(dtype)
            

            if True:
                fuse_R1P(qlayer, P)
                qlayer.register_buffer('P', P)
                if i==0:
                    qlayer.pre_PCA_enable = True
                if i==len(layers)-1:
                    qlayer.post_PCA_enable = True

            if args.train_o_scale == False:
                if args.vo_PCA_init:
                    vo_rotation = PCA_rotation(qlayer.self_attn.o_proj, args.kv_group, args.norm_instruction)
                    vo_rotation.inter_repeat = args.kv_group
                    qlayer.self_attn.o_proj.weight.data = vo_rotation(qlayer.self_attn.o_proj.weight.data)
                    vo_rotation.inter_repeat=1
                    qlayer.self_attn.v_proj.weight.data = vo_rotation(qlayer.self_attn.v_proj.weight.data.T).T
                    del vo_rotation
                
                fuse_R3(qlayer, args, dev)
                
                fuse_R4(qlayer, args, dev)
                qlayer.mlp.Rotation4_enable = True
                
            layers[i] = qlayer.to("cpu")
            del qlayer
    
    fp_outs = fp_outs.to('cpu')
    fp_inps = fp_inps.to('cpu')
    quant_inps = quant_inps.to('cpu')


    ########### Start quantize process ############
   
    for i in range(len(layers)):
        ###### prepare datasets
        logger.info(f"=== Start quantize layer {i} ===")
        qlayer = layers[i].to(dev)
        set_quant_state(qlayer,act_quant=False) 
        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    for j in range(args.nsamples//args.batch_size):
                        index = j * args.batch_size
                        fp_outs[index:index+args.batch_size,] = qlayer(fp_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                        if args.aug_loss:
                            fp_inps_2[index:index+args.batch_size,] = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
        
        ###### init cor-bias
        logger.info(f"=== Prepared quantize layer {i} ===")
        
        for m in qlayer.modules():
            if type(m) == nn.Linear:
                m.weight.requires_grad_(False)

        if args.bias_init =="means":
            print("Doing Bias Init...")
            with torch.no_grad():
                act_disturb = get_act_means(qlayer, fp_inps, 8, 4,['q_proj', 'o_proj', 'up_proj', 'down_proj'],attention_mask=attention_mask,position_ids=position_ids)
                qlayer.self_attn.qbias1.data = act_disturb['q_proj'].mean(dim=0).to(dtype).to(dev)
                qlayer.mlp.qbias3.data = act_disturb['up_proj'].mean(dim=0).to(dtype).to(dev)
                qbias1_init = qlayer.self_attn.qbias1.data.detach()
                qbias2_init = qlayer.mlp.qbias3.data.detach()
            del act_disturb
        
        ######  Stage1: Optimize scale and Rov
        if args.train_o_scale:
            logger.info(f"=== Stage1: Optimize scale and Rov ===")
            with torch.no_grad():
                qlayer.float() 
            replace_o_proj_with_TmpLinear(qlayer)
            replace_down_proj_with_TmpLinear(qlayer)
            replace_v_proj_with_TmpLinear(qlayer)
            qlayer.mlp.down_scale_en = True
            qlayer.mlp.Rotation4_enable = True
            qlayer.self_attn.o_proj.input_trans = True
            qlayer.self_attn.o_proj.dim = args.head_dim
            
            qlayer.self_attn.v_proj.output_trans = True
            qlayer.self_attn.v_proj.dim= args.head_dim
            qlayer.mlp.down_proj.input_trans = True
            qlayer.self_attn.o_quant_enable = True
            qlayer.mlp.down_quant_enable = True
            if args.pre_epochs == 0:
                pre_epochs = args.epochs
            else:
                pre_epochs = args.pre_epochs
            voscale = nn.Parameter((torch.ones(qlayer.self_attn.v_proj.weight.shape[0])*1).to(qlayer.mlp.down_proj.weight.data))
            scale_list = []
            if args.scale_tmask[0]=='1':
                scale_list += [qlayer.self_attn.o_scale2]
            if args.scale_tmask[1]=='1': 
                scale_list += [ qlayer.mlp.down_scale]
            if args.old_scale == False:
                scale_list += [voscale]
            
            if args.vo_PCA_init:
                vo_rotation = PCA_rotation(qlayer.self_attn.o_proj, args.kv_group, args.norm_instruction)
                qlayer.self_attn.o_proj.rotation = vo_rotation
                qlayer.self_attn.v_proj.rotation = copy.deepcopy(vo_rotation)
                qlayer.self_attn.o_proj.rotation.inter_repeat = args.kv_group

            if args.train_vo_rotation:
                vo_rotation = ORTransMatrix(qlayer.self_attn.v_proj.weight.shape[0]//args.head_dim, args.head_dim).to(dev)
                qlayer.self_attn.o_proj.rotation = vo_rotation
                qlayer.self_attn.v_proj.rotation = vo_rotation
                optimizer = torch.optim.AdamW(
                    [{"params":lac_parameters(qlayer),"lr":args.lac_lr},{"params":vo_rotation.parameters(),"lr":args.lvr_lr}, {"params":lab_parameters(qlayer),"lr":args.lab_lr}, {"params":scale_list,"lr":args.lscale_lr}] ,weight_decay=args.wd)
            
            else:
                optimizer = torch.optim.AdamW(
                    [   {"params":scale_list,"lr":args.lscale_lr}] ,weight_decay=args.wd)
        
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = pre_epochs * (args.nsamples // args.batch_size), eta_min=args.lscale_lr * 1e-2)
            loss_scaler = utils.NativeScalerWithGradNormCount() 
            
            for epochs in range(pre_epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size): 
                    index = j * args.batch_size
                    
                    with traincast(device_type='cuda',dtype=args.dtype):
                        if j%10 ==0:
                            tmp = 1/voscale
                            tmp = tmp.reshape(-1,args.head_dim).repeat_interleave(repeats=args.kv_group,dim=0).reshape(-1)
                            qlayer.self_attn.o_proj.find_params(tmp, 1/qlayer.self_attn.o_scale2)
                            qlayer.self_attn.v_proj.find_params(voscale.unsqueeze(1))
                            qlayer.mlp.down_proj.find_params(1/qlayer.mlp.down_scale)
                        
                        tmp = 1/voscale
                        tmp = tmp.reshape(-1,args.head_dim).repeat_interleave(repeats=args.kv_group,dim=0).reshape(-1)

                        qlayer.self_attn.o_proj.quant_tmpweight(tmp, 1/qlayer.self_attn.o_scale2)
                        qlayer.self_attn.v_proj.quant_tmpweight(voscale.unsqueeze(1))
                        qlayer.mlp.down_proj.quant_tmpweight(1/qlayer.mlp.down_scale)

                        quant_out = qlayer(fp_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_outs[index:index+args.batch_size,].to(dev), quant_out)
                        if args.aug_loss: 
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        
                    else:  
                        optimizer.zero_grad()  
                        loss_list.append(loss.detach().cpu())
                        norm = loss_scaler(loss, optimizer,parameters= get_parameters(qlayer)).cpu()
                        scheduler.step()
                        norm_list.append(norm.data)
                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            
            with torch.no_grad():
                replace_o_proj_with_Linear(qlayer)
                replace_down_proj_with_Linear(qlayer)
                replace_v_proj_with_Linear(qlayer)
                qlayer.self_attn.o_quant_enable = False
                qlayer.mlp.down_quant_enable = False
                
                tmp = 1/voscale
                tmp = tmp.reshape(-1,args.head_dim).repeat_interleave(repeats=args.kv_group,dim=0).reshape(-1)
                qlayer.self_attn.o_proj.weight.data = qlayer.self_attn.o_proj.weight.data*tmp
                qlayer.self_attn.v_proj.weight.data = qlayer.self_attn.v_proj.weight.data * voscale.unsqueeze(1)
                qlayer.mlp.down_proj.weight.data = qlayer.mlp.down_proj.weight.data/qlayer.mlp.down_scale.data
                if args.train_vo_rotation:
                    qlayer.self_attn.o_proj.weight.data = vo_rotation(qlayer.self_attn.o_proj.weight.data)
                    qlayer.self_attn.v_proj.weight.data = vo_rotation(qlayer.self_attn.v_proj.weight.data.T).T
                    if qlayer.self_attn.v_proj.bias is not None:
                        qlayer.self_attn.v_proj.bias.data = vo_rotation(qlayer.self_attn.v_proj.bias.data)
                    del vo_rotation
                elif args.vo_PCA_init:
                    qlayer.self_attn.o_proj.weight.data = vo_rotation(qlayer.self_attn.o_proj.weight.data)
                    vo_rotation.inter_repeat=1
                    qlayer.self_attn.v_proj.weight.data = vo_rotation(qlayer.self_attn.v_proj.weight.data.T).T
                    del vo_rotation
                fuse_R3(qlayer, args, dev)
                qlayer.self_attn.o_proj.weight.data = qlayer.self_attn.o_proj.weight.data/qlayer.self_attn.o_scale2
                fuse_R4(qlayer, args, dev)

        with torch.no_grad():
            qlayer = qlayer.to(dtype)
        torch.cuda.empty_cache()



        logger.info(f"=== Stage2: Start GPTQ for Weight quantization ===")
        # in most case we find that use quant_inps in GPTQ process can achieve better acc.
        # TODO: We will replace Weight quantization method to AdaRound to achieve better acc in Future.
        if args.use_fpinps_gptq:
            GPTQ_process(qlayer, args, fp_inps, attention_mask, position_ids, logger, dev)
        else:
            GPTQ_process(qlayer, args, quant_inps, attention_mask, position_ids, logger, dev)
        
        set_quant_state(qlayer, act_quant=True)   # int4
        if args.epochs > 0:
            logger.info(f"=== Stage3: Start learn-able bias-correction, clipping and asymmetric scaling===")
            with torch.no_grad():
                qlayer.float()     
            
            ### setting asymmetic scaling parameters
            asyscale1 = torch.nn.Parameter(torch.ones(args.hidden_dim, device=dev, dtype = torch.float32))  
            asyscale2 = torch.nn.Parameter(torch.ones(args.hidden_dim, device=dev, dtype = torch.float32))
            asyscale3 = torch.nn.Parameter(torch.ones(args.hidden_dim, device=dev, dtype = torch.float32))  
            asyscale4 = torch.nn.Parameter(torch.ones(args.ffn_dim , device=dev, dtype = torch.float32))
            if args.upscale_mask[2]=='1':
                qlayer.self_attn.asyscale3_en = True
            if args.upscale_mask[3]=='1':
                qlayer.mlp.asyscale4_en = True
        
            if args.scale_keep == False:
                optimizer = torch.optim.AdamW(
                    [{"params":lac_parameters(qlayer),"lr":args.lac_lr},{"params":lab_parameters(qlayer),"lr":args.lab_lr},{"params":iter([asyscale1, asyscale2, asyscale3, asyscale4]),"lr":args.lscale_lr}] ,weight_decay=args.wd)
            else:
                optimizer = torch.optim.AdamW(
                    [{"params":lac_parameters(qlayer),"lr":args.lac_lr},{"params":iter(wq_alpha),"lr":args.lac_lr},{"params":lab_parameters(qlayer),"lr":args.lab_lr},{"params":iter([asyscale1, asyscale2, asyscale3, asyscale4]),"lr":args.lscale_lr}] ,weight_decay=args.wd)
                
            scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // args.batch_size), eta_min=args.lscale_lr * 1e-3)
            if args.warmup:
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
                scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
            else:
                scheduler = scheduler_main
            loss_scaler = utils.NativeScalerWithGradNormCount() 
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []

                for j in range(args.nsamples//args.batch_size): 
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast(device_type='cuda', dtype=args.dtype):

                        if args.upscale_mask[0]=='1':
                            qlayer.input_layernorm.weight = 1/asyscale1
                        if args.upscale_mask[1]=='1':
                            qlayer.post_attention_layernorm.weight = 1/asyscale2
                        if args.upscale_mask[2]=='1':
                            qlayer.self_attn.asyscale3 = 1/asyscale3
                        if args.upscale_mask[3]=='1':
                            qlayer.mlp.asyscale4 = 1/asyscale4
                        
                        if args.use_fpinps:
                            quant_out = qlayer(fp_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        else:
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                    

                        loss = loss_func(fp_outs[index:index+args.batch_size,].to(dev), quant_out)
                        if args.aug_loss: 
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,].to(dev), quant_out)
                    
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                    else:  
                        optimizer.zero_grad()  
                        loss_list.append(loss.detach().cpu())
                        norm = loss_scaler(loss, optimizer,parameters= get_parameters(qlayer)).cpu()
                        scheduler.step()
                        norm_list.append(norm.data)
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")        
            
            with torch.no_grad():
                if args.scale_keep:
                    scale_dict, zero_dict = get_Tmp_quantizer_params(qlayer)
                replace_TmpLinear_with_Linear(qlayer)

                if args.upscale_mask[0]=='1':
                    qlayer.input_layernorm.weight = 1/asyscale1
                if args.upscale_mask[1]=='1':
                    qlayer.post_attention_layernorm.weight = 1/asyscale2
                if args.upscale_mask[2]=='1':
                    qlayer.self_attn.asyscale3 = 1/asyscale3
                if args.upscale_mask[3]=='1':
                    qlayer.mlp.asyscale4 = 1/asyscale4
            clear_temp_variable(qlayer)          
            
        qlayer.to(args.dtype)
        clear_temp_variable(qlayer)
        
        del optimizer, asyscale1, asyscale2, asyscale3, asyscale4 
        if not args.disable_realq_replace:
            with torch.no_grad():
                print('replace_linear_with_QLinear')
                replace_linear_with_Quantlinear(qlayer)
        
        torch.cuda.empty_cache()
    
        if args.bias_init =="means":
            print("Doing Bias Compute...")
            with torch.no_grad():
                qbias1_final = qlayer.self_attn.qbias1.data.detach()
                qbias2_final = qlayer.mlp.qbias3.data.detach()
                print('the cosine distance:',(qbias1_final*qbias1_init).sum()/qbias1_final.pow(2).sum().pow(0.5)/qbias1_init.pow(2).sum().pow(0.5) )
                print('the cosine distance:',(qbias2_final*qbias2_init).sum()/qbias2_final.pow(2).sum().pow(0.5)/qbias2_init.pow(2).sum().pow(0.5) )
                print('the L2 distance:', (qbias1_final-qbias1_init).pow(2).sum().pow(0.5)/qbias1_init.pow(2).sum().pow(0.5) )
                print('the L2 distance:', (qbias2_final-qbias2_init).pow(2).sum().pow(0.5)/qbias2_init.pow(2).sum().pow(0.5) )
                Ascale = qlayer.input_layernorm.weight.data
                clip_ratio =  qlayer.self_attn.quantizer1.bound_factor.data
                print('The Ascale mean' ,Ascale.mean(),torch.sigmoid(clip_ratio).mean())

                Ascale = qlayer.self_attn.asyscale3.data
                clip_ratio =  qlayer.self_attn.quantizer2.bound_factor.data
                print('The Ascale mean' ,Ascale.mean(),torch.sigmoid(clip_ratio).mean())

                Ascale = qlayer.post_attention_layernorm.weight.data
                clip_ratio =  qlayer.mlp.quantizer3.bound_factor.data
                print('The Ascale mean' ,Ascale.mean(),torch.sigmoid(clip_ratio).mean())

                Ascale = qlayer.mlp.asyscale4.data
                clip_ratio =  qlayer.mlp.quantizer4.bound_factor.data
                print('The Ascale mean' ,Ascale.mean(),torch.sigmoid(clip_ratio).mean())
        
        if args.epochs>0: 

            if args.qw_ips:
                set_quant_state(qlayer, act_quant=False)
            with torch.no_grad():
                with traincast(device_type='cuda', dtype=args.dtype):
                    for j in range(args.nsamples//args.batch_size): 
                        index = j*args.batch_size
                        quant_inps[index:index+args.batch_size,] = qlayer(quant_inps[index:index+args.batch_size,].to(dtype).to(dev), attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
            if args.qw_ips:
                set_quant_state(qlayer, act_quant=True)
            layers[i] = qlayer.to("cpu")
            fp_inps, fp_outs = fp_outs,fp_inps
        else:
            layers[i] = qlayer.to("cpu")
        
        
    del qlayer
    torch.cuda.empty_cache()
        
    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

