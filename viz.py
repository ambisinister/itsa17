import umap
import umap.plot
import pickle
import numpy as np
from matplotlib import pyplot as plt

train = pickle.load(open('./train.pkl', 'rb'))

feats = []
y = []
for a in train:
    # My training data has one weird entry so I'll just remove it, need to check this later
    if a['rating'] == 'Challenge':
        continue
    bd = a['breakdown']
    sum_measures = sum([x if x >= 0 else 0 for x in bd])
    total_measures = sum([abs(x) for x in bd])
    y.append(int(a['rating']))
    feats.append(np.array([sum_measures, sum_measures/total_measures, int(a['bpm'])]))

mapper = umap.UMAP().fit(feats)
umap.plot.points(mapper, values=np.array(y), theme='fire')
plt.show()
