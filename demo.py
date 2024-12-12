# import os
# import json
# import math
# import torch
# from llava.mm_utils import get_model_name_from_path
# from llava.model.builder import load_pretrained_model


# def split_list(lst, n):
#     """Split a list into n (roughly) equal-sized chunks"""
#     chunk_size = math.ceil(len(lst) / n)  # integer division
#     return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

# num_chunks = 1
# chunk_idx = 0
# model_path = '/data/media/disk/jpf/llava-v1.5-7b'


# # load model 
# model_name = get_model_name_from_path(model_path)
# print(model_name)
# tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
# model.model._diffrate_info["use_diffrate"] = False
# print(model.model._diffrate_info["use_diffrate"])

# model.train(True)
# model.model._diffrate_info["use_diffrate"] = True
# print(model.model._diffrate_info["use_diffrate"])



# # dataloader
# question_file_mini = '/data/media/disk/jpf/llava_v1_5_mix665k_mini.jsonl'
# questions = [json.loads(q) for q in open(os.path.expanduser(question_file_mini), "r")]
# print(len(questions))
# print(questions[0])

# questions = get_chunk(questions, num_chunks, chunk_idx)
# print(len(questions))
# print(questions[0])







import math
import torch

B = 1
N = 2
C = 8

x = torch.randn(B, N, C)
x = abs(x)
print(x)
n = x.norm(dim=-1, keepdim=True)
print(n)
print(x / n)
