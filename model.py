import torch.nn as nn;
import torch.nn.functional as F
import torch;

class TextCNN(nn.Module):
    def __init__(self, embedding, dim=32, n_filters=30, filter_size=None, output_dim=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if filter_size is None:
            filter_size = [3, 4, 5]
        self.embedding = embedding
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=dim,out_channels=n_filters,kernel_size=f) for f in filter_size
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.convs)*n_filters,output_dim);

    def forward(self,vocab):
        embedded = self.embedding(vocab);
        embedded = embedded.permute(0,2,1)#batch,dim,seq_len
        conved = [];
        for conv in self.convs:
            f = F.relu(conv(embedded));
            pooled = F.max_pool1d(f,f.shape[2]).squeeze(2); #batch, n_filters, len(池化后为1，压缩后就降维了)
            conved.append(pooled);
        cat = torch.cat(conved,dim=1)
        dropped = self.dropout(cat)

        return self.fc(dropped) # 全连接


class TextRNN(nn.Module):
    def __init__(self, embedding, dim=32, hidden_dim=64, num_layers=2, output_dim=5, bidirectional=True,
                 dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding

        # 使用双向 LSTM 以获取更好的上下文特征
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)

        # 如果是双向 LSTM，拼接后的维度是 hidden_dim * 2
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, output_dim)

    def forward(self, vocab):
        # embedded shape: (batch_size, seq_len, dim)
        embedded = self.embedding(vocab)

        # LSTM 输出
        output, (hidden, cell) = self.lstm(embedded)

        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # 提取最后一层的隐藏状态
        if self.lstm.bidirectional:
            # 拼接正向最后一个状态和反向最后一个状态
            hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_state = hidden[-1]

        dropped = self.dropout(hidden_state)
        return self.fc(dropped)



class TextTransformer(nn.Module):
    def __init__(self, embedding, dim=32, num_heads=4, num_layers=2, output_dim=5, dropout_rate=0.5, max_len=512, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.dim = dim

        # 可学习的位置编码，应对 Transformer 无法感知位置序列的问题
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_encoder, std=0.02)

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim, output_dim)

    def forward(self, vocab):
        key_padding_mask = (vocab == 0)  # 假设 0 是 <PAD> 的索引

        embedded = self.embedding(vocab)
        seq_len = embedded.size(1)
        embedded = embedded * (self.dim ** 0.5) + self.pos_encoder[:, :seq_len, :]

        transformer_out = self.transformer(embedded, src_key_padding_mask=key_padding_mask)

        mask = (~key_padding_mask).float().unsqueeze(-1)
        sum_pooled = (transformer_out * mask).sum(dim=1)
        actual_lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_pooled / actual_lengths

        dropped = self.dropout(pooled)
        return self.fc(dropped)
