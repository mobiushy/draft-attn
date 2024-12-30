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
    
    num_prop_list = [0, 0] + [de_num for i in range(30)]


    # ==========================================
    # inter = 5
    # stages = 30 // inter
    # prop_num_list = [115 for _ in range(stages+1)]
    # # prop_num_list = [352, 112, 56, 0]
    # last_inter = 30 % inter
    # kept_list = [576, 576]
    # for s in range(stages):
    #     kept_num = kept_list[-1]
    #     prop_num = prop_num_list[s]
    #     if kept_num > prop_num:
    #         stage_list = [kept_num-prop_num for i in range(inter)]
    #     else:
    #         stage_list = [kept_num for i in range(inter)]
    #     kept_list = kept_list + stage_list
    # if last_inter != 0:
    #     kept_num = kept_list[-1]
    #     prop_num = prop_num_list[-1]
    #     if kept_num > prop_num:
    #         last_stage_list = [kept_num-prop_num for i in range(last_inter)]
    #     else:
    #         last_stage_list = [kept_num for i in range(last_inter)]
    #     kept_list = kept_list + last_stage_list
    
    # num_prop_list = []
    # next_kept_list = [kept_list[0]] + kept_list[:-1]
    # for a, b in zip(next_kept_list, kept_list):
    #     num_prop_list.append(a-b)
    # ==========================================
    inter = 5
    pRate = 0.195
    rRate = 1. - pRate
    stages = 30 // inter
    last_inter = 30 % inter
    kept_list = [576, 576]
    for _ in range(stages):
        kept_num = kept_list[-1]
        stage_list = [int(kept_num*rRate) for i in range(inter)]
        kept_list = kept_list + stage_list
    if last_inter != 0:
        kept_num = kept_list[-1]
        last_stage_list = [int(kept_num*rRate) for i in range(last_inter)]
        kept_list = kept_list + last_stage_list
        
    num_prop_list = []
    next_kept_list = [kept_list[0]] + kept_list[:-1]
    for a, b in zip(next_kept_list, kept_list):
        num_prop_list.append(a-b)
    # ===========================================

    # kept_list = [576, 576] + [64 for i in range(30)]
    # kept_list = torch.Tensor([N]).repeat(32)

    print(kept_list)
    print(num_prop_list)
    print(len(kept_list))

    flops = 0
    for num in kept_list:
        N = num
        mhsa_flops = 4*N*C*C + 2*N*N*C
        flops += mhsa_flops
        ffn_flops = 8*N*C*C
        flops += ffn_flops
    
    return flops

flops = calculate_flops(43)
print(flops/1e12)

