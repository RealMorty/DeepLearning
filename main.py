
import torch
import torch.nn.functional as fun
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    batch = 100
    x = np.linspace(-2, 2, batch)[np.newaxis, :]
    noise = np.random.normal(0.0, 0.1, size=(1, batch))
    y = x**2+noise
    return x, y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(1, 5)
        # torch.nn.Linear：建立网络中的全连接层
        self.l2 = torch.nn.Linear(5, 5)
        # self.l3 = torch.nn.Linear(4, 4)
        # self.l4 = torch.nn.Linear(4, 4)
        self.l5 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        # 把原先tensor中的数据按照行优先的顺序排成一个一维的数据
        # 然后按照参数组合成其他维度的tensor
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        # x = torch.sigmoid(self.l3(x))
        # x = torch.sigmoid(self.l4(x))
        # x = fun.leaky_relu(self.l1(x))
        # x = fun.leaky_relu(self.l2(x))
        # x = fun.leaky_relu(self.l3(x))
        # x = fun.leaky_relu(self.l4(x))
        return self.l5(x)


def my_loss_func(loss_out, loss_target):
    loss = (loss_out-loss_target)
    loss = loss.abs().mean()
    return loss


model = Net()
# loss_fn = torch.nn.L1Loss()
# loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.1)


def train(time, data_train, target_train):
    # model.train()
    optimizer.zero_grad()  # set gradient to 0
    outputs = model(data_train)  # input: data
    # print(outputs)
    # loss = loss_fn(outputs, target_train)
    loss = my_loss_func(outputs, target_train)
    # print(target_train)
    loss.backward()  # backward propagation
    optimizer.step()  # update the parameters
    print("[%d] loss: %.6f" % (time+1, loss.item()))


if __name__ == "__main__":
    data_0, target_0 = generate_data()

    [m, bat] = data_0.shape
    permutation = list(np.random.permutation(bat))
    data = data_0[0, permutation]
    target = target_0[0, permutation]
    # 数据打乱

    data_torch = torch.from_numpy(data)
    target_torch = torch.from_numpy(target)
    data_torch = data_torch.view(-1, 1, 1)
    target_torch = target_torch.view(-1, 1)
    data_torch = data_torch.float()
    # 数据转换

    train_data = data_torch[0:50:1, :, :]
    train_target = target_torch[0:50:1, :]
    test_data = data_torch[50:100:1, :, :]
    test_target = target_torch[50:100:1, :]
    # 训练集和测试集划分

    for epoch in range(100):
        train(epoch, train_data, train_target)

    predict_torch = model(test_data)
    predict = predict_torch.detach().numpy()
    # 测试集测试
    plt.scatter(test_data, predict.T, c='r')
    plt.scatter(test_data, test_target, c='b')
    plt.show()
