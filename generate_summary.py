import argparse
from datasets import load_dataset

import torch
from transformers import GPT2Tokenizer

from src.models.sequence.long_conv_lm import ConvLMHeadModel

parser = argparse.ArgumentParser(description='H3 text generation')
parser.add_argument('--dmodel', type=int, default=2048)
parser.add_argument('--nlayer', type=int, default=24)
parser.add_argument('--attn-layer-idx', nargs='+', type=int, default=[8, 16])
parser.add_argument('--rotary_emb_dim', type=int, default=None,
                    help='For rotary embeddings, set to 64. Default is None.')
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--genlen', type=int, default=128)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
args = parser.parse_args()

device = 'cuda'
dtype = torch.float16
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# set seed
torch.random.manual_seed(0)
d_model = args.dmodel
n_layer = args.nlayer
# ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_layer_idx = args.attn_layer_idx
if args.rotary_emb_dim is None:
    attn_cfg = dict(num_heads=args.nheads)
else:
    attn_cfg = dict(num_heads=args.nheads, rotary_emb_dim=args.rotary_emb_dim)
# layer = H3(d_model=d_model, d_state=64, head_dim=1, mode='diag', measure='diag-lin')
layer = {
    '_name_': 'h3',
    'd_state': 64,
    'head_dim': 1,
    'mode': 'diag',
    'measure': 'diag-lin',
}
model = ConvLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                        layer=layer,
                        # ssm_cfg=ssm_cfg,
                        attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                        pad_vocab_size_multiple=8).to(device=device)
if args.ckpt is not None:
    state_dict = torch.load(args.ckpt, map_location=device)
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                      if k.startswith('model.')}
    model.load_state_dict(state_dict)
model.eval()
# Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
# Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
        module.to(dtype=dtype)

# prompt = args.prompt
# load prompt from booksum test set
dataset = load_dataset('booksum', 'test')
prompt = dataset['chapter'][0] + ' \n What is the summary of the given text? \n '

input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)

max_length = input_ids.shape[1] + args.genlen

output_ids = model.generate(input_ids=input_ids, max_length=max_length,
                            return_dict_in_generate=False, output_scores=False,
                            timing=False, top_p=args.top_p, top_k=args.top_k,
                            eos_token_id=tokenizer.eos_token_id)

print(tokenizer.batch_decode(output_ids)[0])
