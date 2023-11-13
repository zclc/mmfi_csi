import torch.nn.functional as F
import torch


def main1():
    t1 = torch.tensor([[[1, 2, 3],
                        [4, 5, 6]],
                       [[10, 20, 30],
                        [40, 50, 60]],
                       [[100, 200, 300],
                        [400, 500, 600]]])

    print(t1.shape)

    t2 = torch.reshape(t1, (2, -1))
    print(t2)
    print(t2.shape)

    s1 = torch.cat([t1[0, :, :], t1[1, :, :], t1[2, :, :]], dim=1)
    print(s1)
    print(s1.shape)


def main2():
    """
    下面例子说明torch.cat()与torch.stack()区别。可以看出，stack()是增加新的维度来完成拼接，不改变原维度上的数据大小。
    cat()是在现有维度上进行数据的增加（改变了现有维度大小），不增加新的维度。
    """
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    print(x)
    print(y)
    print(torch.stack((x, y), 1))
    print(torch.stack((x, y), 1).size())
    print(torch.cat((x, y), 1))
    print(torch.cat((x, y), 1).size())


def main3():
    a = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    b = F.interpolate(a, size=(4, 4), mode='bilinear')
    # 这里的(4,4)指的是将后两个维度放缩成4*4的大小
    print(a)
    print(b)
    print('原数组尺寸:', a.shape)
    print('size采样尺寸:', b.shape)


def main4():
    a = torch.tensor([[1, 4],
                      [5, 8]], dtype=torch.float32)
    a = torch.unsqueeze(a, dim=0)
    a = torch.unsqueeze(a, dim=0)
    b = F.interpolate(a, size=(4, 4), mode='bilinear')

    print(b.shape)
    print(b)


if __name__ == '__main__':
    main1()
