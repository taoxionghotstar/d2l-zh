import random
import torch
# from d2l import torch as d2l


# 构造数据
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X.mv(w) + b + torch.normal(0, 0.01, (num_examples,))
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 生成批次
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X)
    print(y)
    break


# 初始化参数
w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True)
# w = torch.zeros(true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    return X.mv(w) + b


# 定义损失
def squared_loss(y, y_hat):
    return torch.mean((y_hat - y.reshape(y_hat.shape)) ** 2) / 2


def squared_loss_d2l(y, y_hat):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 优化算法
def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


def sgd_d2l(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 模型训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l):f}')


# loss = squared_loss_d2l
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd_d2l([w, b], lr, batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
