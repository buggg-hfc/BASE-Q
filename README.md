## The official code of paper: BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models

### Installation

```bash
conda create -n base_q python=3.12 -y
conda activate base_q
pip install -r requirements.txt
```

### Perform W4A4KV4 Quantization

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model /LLMs/Llama-2-7B --device_map auto --eval_ppl --dtype float16 \
    --wbits 4 --w_sym  --abits 4 --kbits 4 --vbits 4  --true-sequential --act-order --disable_realq_replace \
    --pre_epochs 5 --epochs 5 --nsamples 256 --lac --lac_lr 3e-2 --lvr_lr 3e-3 --lscale_lr 3e-3 --batch_size 2  \
    --use_fpinps --upscale_mask 1111 --Rres_init Hadamard  --train_vo_rotation --train_o_scale \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande
```
Note: You can set `--dtype bfloat16` in some model to have better performance