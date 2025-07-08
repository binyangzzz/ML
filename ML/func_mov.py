import torch
import torch.optim as optim  # 导入优化器模块
import matplotlib.pyplot as plt

# 定义损失函数
def loss_fn(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

# 定义模型
model = torch.nn.Linear(10, 1)

# 定义训练数据
x_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器，学习率为 0.001

# 定义损失函数
loss_list = []

# 开始训练
for epoch in range(10000):
    # 前向传播
    y_pred = model(x_train)

    # 计算损失
    loss = loss_fn(y_train, y_pred)
    loss_list.append(loss.item())

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 展示损失
    if epoch % 10 == 0:    # 10 个epoch绘画一个损失函数点，可以自定义
        print(f"epoch {epoch}: loss {loss.item()}")
        # 更新损失曲线
        plt.cla()
        plt.plot(loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.pause(0.01)

# 绘制损失曲线
plt.show()

