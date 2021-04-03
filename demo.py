import torch


# x = torch.rand(5,3)
# print(x)

# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# y = x + 2
# print(y)
#
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
#
# print(z, out)

def test01():
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)


if __name__ == "__main__":
    test01()