import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,train,dim=32):
        super().__init__()
        # 将 self.train 改为 self.train_data 防止与内置方法冲突
        self.train_data = train
        self.words = {"<PAD>":0,"<UNK>":1}
        self.dim = dim
        self.embedding_dim = dim
        self.vocabulary()

    def vocabulary(self):
        # 这里也要同步修改为 self.train_data
        for i in self.train_data:
            for j in i.split():
                if(j not in self.words):
                    self.words[j] = len(self.words)

        self.embedding_layer = nn.Embedding(num_embeddings=len(self.words), embedding_dim=self.dim,padding_idx=0)

    def serialize(self,data):
        max_length = max(len(s.split()) for s in data)
        self.encode_data = []
        for sentence in data:
            sentence = sentence.split()
            vec = []
            for word in sentence:
                vec.append(self.words.get(word,1))
            vec += [0] * (max_length - len(vec))
            self.encode_data.append(vec)

        self.tensor = torch.LongTensor(self.encode_data)
        return self.tensor

    def forward(self,tensor):
        return self.embedding_layer(tensor)

    def get_embedding_layer(self):
        return self.embedding_layer