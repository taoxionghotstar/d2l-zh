import torch
from torch.utils import data
from torch import nn


# 构造数据
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X.mv(w) + b + torch.normal(0, 0.01, (num_examples,))
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


# 网络结构
net = nn.Sequential(nn.Linear(2, 1))

# 参数初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失
loss = nn.MSELoss()

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('w的估计误差：', true_b - b)
