# This file is modified from https://github.com/OpenGVLab/OmniQuant/main.py

import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from pprint import pprint

import torch.nn as nn
from quantize.baseq import baseq
from tqdm import tqdm
import utils
from pathlib import Path

import pdb

from datasets import Dataset

torch.backends.cudnn.benchmark = True

net_choices = [
    "Qwen2.5-3B",
    "Qwen2.5-14B",
    "Qwen2.5-32B",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-3-8B",
    "Llama-3-70B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Llama-3.2-1B",
    "Llama-3.2-3B"
]


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    
    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        lm.model = lm.model.to(lm.device)
   
    if args.eval_ppl:
        for dataset in ["wikitext2" ]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.cache_name}_{dataset}_all.cache'
            if os.path.exists(cache_testloader) :
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)

            testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            ########## the ppl computing process:
            ########## 
            if args.limit>0 :
                nsamples = args.limit
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                with torch.amp.autocast(device_type='cuda', dtype=args.dtype):
                    if  "llama" in args.net.lower() or "qwen" in args.net.lower():
                        outputs = lm.model.model(batch)
                    hidden_states = outputs[0]
                    logits = lm.model.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                        :, 1:
                    ].to(lm.model.lm_head.weight.device)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)).float(),
                        shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * lm.seqlen
                    nlls.append(neg_log_likelihood)
                    
            ############ 
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.tasks != "":
        
        with torch.amp.autocast(device_type='cuda'):
            hflm = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer,batch_size=1)
            results = evaluator.simple_evaluate(
                hflm,
                tasks=args.tasks.split(','),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit == -1 else args.limit,
            )

        
        logger.info(make_table(results))

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results['results'].items()}
        mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in results['results'].items()}
        mean_std_val =round(sum(std_vals.values()) / len(std_vals.values()), 4) 
        metric_vals['acc_avg'] = mean_acc_val
        results['results']['AVERAGE'] = {
            "acc,none":mean_acc_val,
            "acc_stderr,none":mean_std_val
        }
        logger.info(make_table(results))
            
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--scale_load_dir", default=None, type=str, help="direction for loading clipping paras")

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2"],
        help="Where to extract calibration data from.",
    )

    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for bias and asymmetric scaling")
    parser.add_argument("--pre_epochs", type=int, default=0, help="Pre Training epochs for scaling and rotation_ov")
    
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)

    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--kbits", type=int, default=4)
    parser.add_argument("--vbits", type=int, default=4)

    parser.add_argument("--lab_lr", type=float, default=2e-3)
    parser.add_argument("--lscale_lr", type=float, default=5e-4)
    parser.add_argument("--lac_lr", type=float, default=1e-2)
    parser.add_argument("--lvr_lr", type=float, default=5e-3)

    parser.add_argument("--wd", type=float, default=0)

    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--lac",default=False, action="store_true",help="activate learnable activation clipping")

    parser.add_argument("--k_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--v_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--a_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--r4_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    #

  
    parser.add_argument("--upscale_mask", type=str, default="1111")
    parser.add_argument("--quant_mask", type=str, default="1111")
    parser.add_argument("--scale_tmask", type=str, default="11")
    parser.add_argument("--scale_imask", type=str, default="11")
    

    parser.add_argument("--k_bias", default=False, action="store_true")
    parser.add_argument("--v_bias", default=False, action="store_true")
    parser.add_argument("--scale_keep", default=False, action="store_true")
    parser.add_argument("--qw_ips", default=False, action="store_true")
    parser.add_argument("--eval_KVnoq", default=False, action="store_true")
    parser.add_argument("--use_fpinps", default=False, action="store_true")
    parser.add_argument("--use_fpinps_gptq", default=False, action="store_true")
    parser.add_argument("--warmup", default=False, action="store_true")
    parser.add_argument("--train_o_scale", default=False, action="store_true")
    parser.add_argument("--train_vo_rotation", default=False, action="store_true")
    parser.add_argument("--old_scale", default=False, action="store_true")
    parser.add_argument("--vo_PCA_init", default=False, action="store_true")
    parser.add_argument("--Rres_init", default='Hadamard', type=str, choices=["Hadamard","PCA","Spinquant"])
    parser.add_argument("--Rres_init_path", default='None', type=str)
    parser.add_argument("--bias_init", default='zeros', type=str, choices=["zeros","means"])
    parser.add_argument("--norm_instruction", type=str, default="ori",choices=["ori","orib","hnorm", "norm"])
    parser.add_argument("--dtype", type=str, default="float16",choices=["float16","bfloat16"])
    parser.add_argument("--disable_realq_replace", default=False, action="store_true")
    parser.add_argument("--disable_rotation", default=False, action="store_true")
    parser.add_argument("--only_eval", default=False, action="store_true")
    parser.add_argument("--pre_eval", default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager"],
        help="attention implementation that the model works with",
    )
    


    parser.add_argument('--mse', default=True,  action='store_false', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument( '--Use_RTN', default=False, action='store_true', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--w_sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.' )
    parser.add_argument( '--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument(  '--static-groups', action='store_true', help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')
    
    parser.add_argument( '--info', action='store_true', help='Show info')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if (args.wbits<16 and args.wbits>8) or (args.abits<16 and args.abits>8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    if args.dtype =="float16":
        args.dtype = torch.float16
    if args.dtype =="bfloat16":
        args.dtype = torch.bfloat16
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    args.hidden_dim = lm.model.config.hidden_size
    args.kv_group = lm.model.config.num_attention_heads // lm.model.config.num_key_value_heads
    args.ffn_dim = lm.model.config.intermediate_size
    args.heads = lm.model.config.num_attention_heads
    args.head_dim = args.hidden_dim//args.heads
    
    
    for param in lm.model.parameters():
        param.requires_grad = False
    
    args.q_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    args.act_quant_params = {
        "n_bits": args.abits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "lac":args.lac,
    }
    
    args.R2_quant_params = {
        "n_bits": args.kbits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method":args.k_quant_method,
        "quant_bias": args.k_bias,
        "lac":args.lac,
    }
    args.R3_quant_params = {
        "n_bits": args.vbits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method":args.v_quant_method,
        "quant_bias": args.v_bias,
        "lac":args.lac,
    }



    # quantization
    if 'llama-2' in args.net.lower():
        args.cache_name = 'llama-2'
    elif 'llama-3' in args.net.lower():
        args.cache_name = 'llama-3'
    elif 'qwen2.5' in args.net.lower():
        args.cache_name = 'qwen2.5'

    if args.pre_eval:
        evaluate(lm, args,logger)

    if not args.only_eval:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.cache_name}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader, weights_only=False)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        
        baseq(
            lm,
            args,
            dataloader,
            logger,
        )

        logger.info(time.time() - tick)
    
    evaluate(lm, args,logger)

if __name__ == "__main__":
    print(sys.argv)
    main()
