import torch
from torch import nn
from torch.nn import MaxPool2d, AvgPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))


class Tuidui(nn.Module):
    def __init__(self):
        super(Tuidui, self).__init__()
        self.maxpool1 = AvgPool2d(kernel_size=(1, 2), ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


input2 = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
print(input2)

tudui = Tuidui()
output = tudui(input2)
output.squeeze_()
print(output)

