import pandas as pd
from sklearn.model_selection import train_test_split
from embedding import Embedding
from model import TextCNN, TextRNN, TextTransformer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dim = 64

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print(device)
train_df = pd.read_csv("../new_train.tsv", sep='\t', header=None, names=['C1', 'C2'])
test_df = pd.read_csv("../new_test.tsv", sep='\t', header=None, names=['C1', 'C2'])

x = train_df['C1'].astype(str)
# 修复：加上 .values 将 Series 转换为 Numpy 数组
y = train_df['C2'].values

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_test = test_df["C1"].astype(str)
# 修复：加上 .values
y_test = test_df["C2"].values

embedding = Embedding(x_train,dim=dim).to(device)

x_train = embedding.serialize(x_train).to(device)
x_val = embedding.serialize(x_val).to(device)
x_test = embedding.serialize(x_test).to(device)

y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)


def train_model(net, train_x, train_y, val_x, val_y, epochs=50, batch_size=64):
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3,weight_decay=1e-3)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        net.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            # 数据已经在device上，不需要再移动
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_train_loss = total_loss / len(train_loader)
            net.eval()
            with torch.no_grad():
                val_outputs = net(val_x)
                val_loss = criterion(val_outputs, val_y)
                print(f"Epoch:{epoch + 1}/{epochs}, train_loss:{avg_train_loss:.4f}, val_loss:{val_loss:.4f}")


def evaluate(model, data_x, data_y_true):
    model.eval()
    with torch.no_grad():

        outputs = model(data_x)
        _, predicted = torch.max(outputs, 1)

        correct = (predicted == data_y_true).sum().item()
        total = data_y_true.size(0)

        accuracy = correct / total
        return accuracy



model = TextRNN(embedding, dim=dim, output_dim=5).to(device)
train_model(model, x_train, y_train_t, x_val, y_val_t, epochs=50, batch_size=50)
print(evaluate(model, x_val, y_val_t))
#
# plt.plot(lr, acc, marker='o', linestyle='-', color='b')
# plt.xlabel('lr')
# plt.ylabel('accuracy')
# plt.title('lr')
# plt.savefig("graph_2/overview.png")
