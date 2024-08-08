import torch
import numpy as np
import matplotlib.pyplot as plt

def get_barrier(x: torch.Tensor, min_val: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    print(x.item())
    sum_ = torch.sum(x)
    diff = limit - sum_
    return torch.nn.ReLU()(-diff) -torch.log(torch.max(diff,min_val))+torch.exp(-diff)

if __name__ == "__main__":
    min_val = torch.Tensor([1e-6])
    limit = torch.Tensor([5.0])
    lower_bound = True
    xs = []
    ys = []
    for p in np.arange(-15,20.01,0.2):
        x = torch.Tensor([p])
        xs.append(x)
        ys.append(get_barrier(x, min_val, limit).item())
    f,ax = plt.subplots()
    ax.plot(xs,ys,marker = "|")
    plt.pause(1000)



