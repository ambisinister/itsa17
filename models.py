import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import math
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, hidden=128):
        super(RNN, self).__init__()
        self.hiddensize = hidden
        self.to_hidden = nn.Linear(1+hidden, hidden)
        self.out = nn.Linear(1+hidden, 1)

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        x = x.to("cuda")

        conc = torch.cat((x, hidden), 1)
        hidden = self.to_hidden(conc)
        out = self.out(conc)
        
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hiddensize).cuda()

class RNNClassifier():
    def __init__(self, learning_rate=3e-4, epochs=100):
        self.net = RNN()
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr = learning_rate)
        
        self.net.to("cuda")

    def predict(self, X):
        X = self.process_input([X])
        x = X[0]
        x = x.to("cuda")
        hidden = self.net.init_hidden()        

        for i in range(x.size()[0]):
            block = x[i].unsqueeze(0)
            out, hidden = self.net(block, hidden)

        return out.detach().cpu().numpy()[0][0]

    def process_input(self, x, labels=False):
        if labels:
            labels = [torch.FloatTensor([int(a)]) for a in x]
            return labels
        else:
            breakdowns = [np.array([*a[0], int(a[1])]).astype(float) for a in x]
            
            #pad with zeros, unneeded usually
            #max_size = 400
            #breakdowns = [[*a, *[0 for _ in range(max_size-len(a))]] for a in breakdowns]
            #breakdowns = np.array(breakdowns).astype(float)

            return [torch.FloatTensor(a) for a in breakdowns]

    def set_eval(self):
        self.net.eval()

    def learn(self, inp_x, inp_y):
        hidden = self.net.init_hidden()
        self.net.zero_grad()
        criterion = self.criterion

        for i in range(inp_x.size()[0]):
            block = inp_x[i].unsqueeze(0)
            out, hidden = self.net(block, hidden)

        loss = criterion(out[0][0], inp_y[0])
        loss.backward()
        self.optim.step()

        return out, loss.item()

    def fit(self, train_X, train_Y):
        train_X = self.process_input(train_X) 
        train_Y = self.process_input(train_Y, labels=True)
        criterion = self.criterion

        for epoch in range(self.epochs):
            current_loss = 0

            for i, (inp, lab) in enumerate(zip(train_X, train_Y)):
                if i % 500 == 0:
                    print(f"epoch {epoch} {i}/{len(train_X)}")
                inp = inp.to("cuda")
                lab = lab.to("cuda")

                out, loss = self.learn(inp, lab)
                current_loss += loss

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {current_loss}")
        
        


class LookupModel():
    def __init__(self, use_max=True):
        self.use_max = use_max
        self.measures = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        self.bpms = list(np.arange(100, 351, 10))
        self.table = [[8.5, 8.75, 9.5, 9.75, 10, 10.5, 10.75, 11, 11.5, 11.5, 11.75, 12, 12.5],
                      [8.75, 9.5, 9.75, 10, 10.5, 10.75, 11.5, 11.75, 12, 12.5, 12.75, 13, 13.5],
                      [9.5, 9.75, 10.5, 10.75, 11.5, 11.75, 12.5, 12.5, 12.75, 13, 13.5, 13.75, 13.75],
                      [9.75, 10, 10.5, 10.75, 11.5, 12, 12.5, 13, 13.5, 13.75, 14, 14.5, 14.75],
                      [10.5, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 13.75, 14, 14.5, 14.75, 15.5],
                      [10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 15.75, 16.5],
                      [11, 11.5, 12.5, 12.75, 13.5, 13.75, 14, 14.5, 15.5, 15.75, 16, 16.5, 16.75],
                      [11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15.5, 15.75, 16, 16.5, 17, 17.5],
                      [12.5, 12.75, 13, 13.5, 13.75, 14.5, 15.5, 15.75, 16.5, 16.75, 17, 17.5, 18],
                      [12.75, 13.5, 13.75, 14, 14.5, 15.5, 15.75, 16.5, 17, 17.5, 18.5, 18.75, 19.5],
                      [13.5, 13.75, 14, 14.5, 15.5, 15.75, 16.5, 17.5, 17.75, 18.5, 19.5, 19.75, 20.5],
                      [13.75, 14.5, 14.75, 15.5, 16, 16.5, 17.5, 18, 18.5, 19.5, 20.5, 20.75, 21.5],
                      [14.5, 14.75, 15.5, 16.5, 16.75, 17.5, 18.5, 19.5, 19.75, 20.5, 21.5, 22, 22.5],
                      [14.75, 15.5, 16, 16.5, 17.5, 18.5, 19.5, 20.5, 20.75, 21.5, 22.5, 22.75, 23.5],
                      [15.75, 16.5, 16.75, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 23.75, 24.5, 25],
                      [16.5, 17.5, 18, 18.5, 19.75, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 25.75, 26.5],
                      [17.5, 18.5, 19, 19.75, 21, 22, 23, 23.75, 24.5, 25.5, 26.5, 26.75, 27.5],
                      [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25, 25.75, 26.5, 27, 27.5, 28],
                      [19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26, 26.75, 27.5, 27.75, 28.5, 29.5],
                      [20.5, 21.5, 22.5, 23.5, 24.5, 25, 26, 27.5, 27.75, 28.5, 28.75, 29.5, 30.5],
                      [21.5, 22.5, 23.5, 24.5, 24.75, 25.5, 26.5, 27.75, 28.5, 29.5, 30, 30.5, 31.5],
                      [22, 23, 24, 24.75, 25.5, 26.75, 27.5, 28.5, 29.5, 29.75, 30.5, 31, 31.75],
                      [22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29, 30.5, 30.75, 31.5, 32, 32.5],
                      [23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29, 30.5, 31.5, 31.75, 32.5, 32.75, 33.5],
                      [24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30, 31.5, 31.75, 32.5, 32.75, 33.5, 34.5],
                      [25.5, 26.5, 27.5, 28, 28.75, 29.5, 30.5, 31.75, 32.5, 33, 33.5, 34.5, 35.5]
                      ]
    def lookup(self, measures, bpm):
        # convert to int for comparisons
        measures, bpm = int(measures), int(bpm)
        
        # if in table, use table value
        if measures in self.measures and bpm in self.bpms:
            return self.table[self.bpms.index(bpm)][self.measures.index(measures)]

        # if not in table, interpolate along the corners

        # Snap to edges of table
        if measures > 512:
            return self.lookup(512, bpm)
        if bpm > 350:
            return self.lookup(measures, 350)
        if measures < 8:
            return self.lookup(8, bpm)
        if bpm < 100:
            return self.lookup(measures, 100)

        # Get four corners for interp
        def binsearch(ls, x):
            low = 0
            high = len(ls)-1

            while low<high:
                mid = (low+high)//2
                if ls[mid] == x:
                    return ls[mid], ls[mid]
                if low == mid:
                    return ls[low], ls[high]
                
                if x < ls[mid]:
                    high = mid
                else:
                    low = mid

        def lin_interp(x1, x2, z1, z2, x):
            # no change in x div 0 case
            if (x2-x1) == 0:
                return (z2+z1)/2
            grad = (z2-z1)/(x2-x1)
            dx = x - x1
            return z1 + (dx * grad)
            
        lowbpm, highbpm = binsearch(self.bpms, bpm)
        lowmea, highmea = binsearch(self.measures, measures)

        # bilinear interpolation
        r1 = lin_interp(lowmea, highmea,
                        self.lookup(lowmea, lowbpm), self.lookup(highmea, lowbpm), measures)
        r2 = lin_interp(lowmea, highmea,
                        self.lookup(lowmea, highbpm), self.lookup(highmea, highbpm), measures)
        return lin_interp(lowbpm, highbpm, r1, r2, bpm)

    def predict(self, song):
        [breakdown, bpm] = song
        if self.use_max:
            return math.floor(self.lookup(max(breakdown), bpm))
        else:
            sum_measures = sum([x for x in breakdown if x > 0])
            return math.floor(self.lookup(sum_measures, bpm))

    # Nothing to be done
    def fit(self, train_x, train_y):
        pass


class StreamClassifier():
    def __init__(self, model=LinearRegression()):
        self.net = model

    def _transform_breakdown(self, song):
        bd, bpm = song
        sum_measures = sum([x if x >= 0 else 0 for x in bd])
        total_measures = sum([abs(x) for x in bd])
        return np.array([sum_measures, sum_measures/total_measures, bpm])

    def fit(self, train_x, train_y):
        train_x = np.array([self._transform_breakdown(x) for x in train_x], dtype=object)
        train_y = [int(y) for y in train_y]
        self.net.fit(train_x, train_y)

    def predict(self, inp):
        if len(np.shape(inp)) == 1:
            inp = [inp]
        trans_inp = np.array([self._transform_breakdown(x) for x in inp], dtype=object)
        return self.net.predict(trans_inp)

    
if __name__ == '__main__':

    a = LookupModel()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(8,512,2)
    Y = np.arange(100,350,2)
    Z = np.array([[a.lookup(x,y) for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)

    ax.set_xlabel('Measures of Stream')
    ax.set_ylabel('BPM')
    ax.set_zlabel('Rating')
    ax.set_title('BPM + Stream vs Rating - Lookup Table')

    print(X)
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(Z))


    Axes3D.plot_surface(ax,X,Y,Z)
    plt.show()
