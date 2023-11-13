import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'using {device} device!')

def main1():
    t1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    t1.to
    print(t1)
    t2 = torch.transpose(t1, 0,1)
    print(t2)


if __name__ == '__main__':
    main1()