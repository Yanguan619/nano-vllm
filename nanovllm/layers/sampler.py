import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    # @torch.compile
    def forward_simple(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        logits = logits.div_(temperatures.unsqueeze(dim=1)) # 广播到logits的维度,对logits的元素进行缩放
        # 对logits采样得到token的索引
        # 若采样为1,则得到一个token的索引,若采样为batch_size,则得到batch_size个token的索引
        # 索引前后按照概率分布大小排序
        probs = torch.softmax(logits, dim=-1) # 得到token的概率分布
        sample_tokens = torch.multinomial(probs, num_samples=1)
        return sample_tokens


    # @torch.compile
    def forward_Gumbel_Max(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        logits = logits.div_(temperatures.unsqueeze(dim=1))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        sample_tokens = torch.argmax(logits + gumbel_noise, dim=-1)
        return sample_tokens


if __name__ == "__main__":
    sampler = Sampler()
    logits = torch.Tensor([[0.1,0.2, 0.9],[0.1,0.2,   0.9]])
    temperatures = torch.Tensor([1.0, 0.9])
    print(f'{logits=} {temperatures=}')
    print(sampler.forward(logits, temperatures))
    print(sampler.forward_simple(logits, temperatures))
    print(sampler.forward_Gumbel_Max(logits, temperatures))
