import torch.nn as nn
 
class RatingLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.softmax=nn.Softmax()
        
    def forward(self, x, hidden):
        batch_size=x.size()
        
        embed=self.embedding(x)
        lstm_out, hidden=self.lstm(embed, hidden)
        
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out=self.fc1(lstm_out)
        out=self.fc2(out)
        out=self.fc3(out)
        sm_out=self.softmax(out)
        
        sm_out=sm_out.view(batch_size, -1)
        sm_out=sm_out[:, -1]
        
        return sm_out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        
        return hidden

net = RatingLSTM(vocab_size=1024, output_size=1, embedding_dim=200, hidden_dim=128, n_layers=3)

#write training functions later
#https://bhadreshpsavani.medium.com/tutorial-on-sentimental-analysis-using-pytorch-b1431306a2d7
