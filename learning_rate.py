import argparse
import os
import torch
from torch import nn
import json
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from torch import inf


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        # qs = line["text"]
        qs = line["conversations"][0]["value"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        labels = input_ids
        
        return input_ids, image_tensor, labels

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def rest_flops(tk_num, layer_idx): 
    res_flops = 0
    # output projection + FFN
    res_flops += (tk_num * 4096**2 + 3 * 11008 * 4096 * tk_num)
    # flops for each remaining layer
    left_layer = 31 - layer_idx
    res_flops += ((tk_num * 4096 * 4096 * 3 + tk_num * tk_num * 4096 + tk_num * 4096 * tk_num + tk_num*4096*4096) + (4096*11008*tk_num * 3)) * left_layer
    return res_flops


def initialize_model_and_loader(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        
    # data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=1)
    
    import transformers
    from llava.train.train import DataArguments, LazySupervisedDataset
    parser = transformers.HfArgumentParser((DataArguments))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args.image_processor = image_processor
    data_args.data_path = '/data/media/disk/jpf/llava_sft_data.json'
    data_args.image_folder = '/data/media/disk/jpf/llava_data'
    data_args.mm_use_im_start_end = False
    data_args.is_multimodal = True

    dataset = LazySupervisedDataset(data_args.data_path, tokenizer, data_args)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    return model, data_loader, questions


def calculate_origin_flops():
    return ((576 * 4096 * 4096 * 3 + 576 * 576 * 4096 + 576 * 4096 * 576 + 576 * 4096 * 4096) + (4096 * 11008 * 576 * 3)) * 32


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                # norm = ampscaler_get_grad_norm(parameters)
                norm = ampscaler_get_grad_norm(parameters, inf)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def compute_attention_weights(model, input_ids, image_tensor, labels, device):
    sys_len = 36
    left_vtoken_num = 576
    weightsum_cross = []
    weightsum_self = []
        
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(device=device, non_blocking=True)
    image_tensor = image_tensor.to(device=device).to(dtype=torch.float16)
    attention_mask = attention_mask.to(device=device)


    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=None,
        labels=labels,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        images=image_tensor,
        return_dict=True    
    )

    for i in range(32):
        attn_weights = outputs.attentions[i].max(dim=1).values.squeeze()
        weightsum_cross.append( attn_weights[sys_len + left_vtoken_num:, sys_len:sys_len+left_vtoken_num].sum() / (input_ids.shape[1] - sys_len - left_vtoken_num) )
        self_part = attn_weights[sys_len: sys_len+left_vtoken_num, sys_len:sys_len+left_vtoken_num].sum()
        weightsum_self.append(self_part/max(left_vtoken_num,1))
    
    for ii in range(len(weightsum_cross)):
        weightsum_cross[ii] = weightsum_cross[ii].to('cuda:0')
        weightsum_self[ii] = weightsum_self[ii].to('cuda:0')
    
    weightsum_cross = torch.stack(weightsum_cross)
    weightsum_self = torch.stack(weightsum_self)
    
    return weightsum_cross, weightsum_self, outputs

def statistical_analysis(args):
    # Model
    print("Loading Model...")
    model, data_loader, questions = initialize_model_and_loader(args)
    model.train()

    for data in tqdm(data_loader):
        for k in data:
            print(k, data[k].shape)
        break


    cnt = 0
    for inputs in tqdm(data_loader):
        # model.model._diffrate_info["use_diffrate"] = False
        # tar_cross, tar_self, _ = compute_attention_weights(model, input_ids, image_tensor, inputs['labels'], device='cuda')

        # model.model._diffrate_info["use_diffrate"] = True
        # cur_cross, cur_self, outputs = compute_attention_weights(model, input_ids, image_tensor, inputs['labels'], device='cuda')

        # criterion = nn.CrossEntropyLoss()
        # loss_ce = criterion(cur_cross, tar_cross)


        # model.model._diffrate_info["use_diffrate"] = False

        device = 'cuda'
        attention_mask = torch.ones_like(inputs['input_ids'])
        input_ids = inputs['input_ids'].to(device=device, non_blocking=True)
        image_tensor = inputs['image'].to(device=device).to(dtype=torch.float16)
        attention_mask = attention_mask.to(device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=None,
            labels=inputs['labels'],
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            images=image_tensor,
            return_dict=True    
        )


        flops = model.model.calculate_flops_training()


        optimizer = torch.optim.AdamW(model.model.arch_parameters(), lr=0.01, weight_decay=0)
        # print(optimizer.param_groups[0]['params'])
        loss_scaler = NativeScalerWithGradNormCount()
        criterion = nn.CrossEntropyLoss()

        with torch.cuda.amp.autocast():
            # loss_ce = criterion(cur_cross, tar_cross)
            loss_ce = torch.Tensor([1]).to('cuda:0')
            # loss_ce = outputs['logits'].mean()
            # loss_ce = outputs['attentions'][-1].mean()*100
            # loss_ce = criterion(exps, tars)
            # loss_ce = outputs["loss"]
            
            loss_flops = ((flops/1e12)-1.5)**2
            loss = loss_ce + 5*loss_flops
        
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=None,
                                parameters=model.model.arch_parameters(),
                                create_graph=is_second_order)
        

        print('loss_ce: ', loss_ce.item())
        print('loss_flops: ', loss_flops.item())
        print('grad_norm: ', grad_norm)
        print('flops: ', (model.model.calculate_flops_inference()/1e12).item())


        print(model.model.get_kept_num())
        
        if cnt == 50:
            break

        cnt += 1

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/media/disk/jpf/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/media/disk/jpf/llava_data")
    parser.add_argument("--question-file", type=str, default="/data/media/disk/jpf/llava_v1_5_mix665k_mini.jsonl") # 
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--reduction_ratio", type=int, default=0.6)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    args = parser.parse_args()

    
    statistical_analysis(args)





