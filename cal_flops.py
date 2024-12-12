import torch

def calculate_flops(de_num):
    N = 576
    C = 4096
    de_list = torch.arange(N, 0, -de_num)
    pre_list = torch.Tensor([N])
    if len(de_list) < 31:
        suf_list = de_list[-1].repeat(31-len(de_list))
        kept_list = torch.cat([pre_list, de_list, suf_list])
    else:
        de_list = de_list[:31]
        kept_list = torch.cat([pre_list, de_list])


    kept_list = [576, 576] + [128 for i in range(30)]
    # kept_list = torch.Tensor([N]).repeat(32)

    print(kept_list)
    print(len(kept_list))

    flops = 0
    for num in kept_list:
        N = num
        mhsa_flops = 4*N*C*C + 2*N*N*C
        flops += mhsa_flops
        ffn_flops = 8*N*C*C
        flops += ffn_flops
    
    return flops

flops = calculate_flops(39)
print(flops/1e12)

