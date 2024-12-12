import torch


def generate_mask(num_query, num_kv, dtype, device):
    mask = torch.full(
        (1, 1, num_query, num_kv), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )
    assert num_query <= num_kv
    mask[0,0,:,-num_query:].triu_(diagonal=1)
    mask[0,0,:,:-num_query].fill_(0)
    mask = mask.type(dtype)
    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')
    return mask