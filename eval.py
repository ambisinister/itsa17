import pickle
import numpy as np
from models import LookupModel, StreamClassifier, RNNClassifier
from matplotlib import pyplot as plt

#models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Kill warnings for inference
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

if __name__ == '__main__':
    train = pickle.load(open('./train.pkl', 'rb'))
    val = pickle.load(open('./val.pkl', 'rb'))    
    test = pickle.load(open('./test.pkl', 'rb'))

    # Clean Train so nothing odd happens
    print(len(train))
    for x in train:
        try:
            _ = int(x['rating'])
        except:
            train.remove(x)
    print(len(train))
    
    train_X = [[x['breakdown'], x['bpm']] for x in train]
    train_y = [x['rating'] for x in train]
    val_X = [[x['breakdown'], x['bpm']] for x in val]
    val_y = [x['rating'] for x in val]
    test_X = [[x['breakdown'], x['bpm']] for x in test]
    test_y = [x['rating'] for x in test]

    ml_models = [
        MLPClassifier(max_iter=1000),
        KNeighborsClassifier(3),
        SVC(kernel='linear', C=0.025),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GaussianNB()
    ]
    streamclassifiers = [StreamClassifier(x) for x in ml_models]
    models = [LookupModel(),
              LookupModel(use_max=False),
              StreamClassifier(),
              *streamclassifiers,
              RNNClassifier()]
    titles = ['Lookup Table Interpolation',
              'Lookup Table Interpolation (Sum of Total Stream)',
              'Linear Regression',
              'Multilayer Perceptron',
              'kNN',
              'SVM Classifier',
              'Decision Tree',
              'Random Forest',
              'Naive Bayes',
              'Recurrent Neural Network']

    for model, title in zip(models, titles):
        model.fit(train_X, train_y)

        if title == 'Recurrent Neural Network':
            model.set_eval()

        #Evaluate
        def eval_model(mod, x, y):
            ratings = []
            preds = []

            for song, rating in zip(x, y):
                pred = mod.predict(song)
                ratings.append(rating)
                preds.append(pred)

            ratings = [int(x) for x in ratings]

            accuracy = sum([x==int(y) for x,y in zip(ratings, preds)])/len(ratings)
            print(f"{title} accuracy: {accuracy}")
            mae = np.mean([abs(x-y) for x,y in zip(ratings, preds)])
            print(f"{title} Mean Absolute Error: {mae}")
            return ratings, preds

        print("Reclassification Accuracy:")
        _, _ = eval_model(model, train_X, train_y)
        print("Validation Accuracy:")
        ratings, preds = eval_model(model, val_X, val_y)

        plt.title(f'{title}')
        plt.scatter(ratings, preds, alpha=.25)
        plt.plot(ratings, ratings, color='red')
        plt.xlabel('ratings')
        plt.ylabel('predictions')
        plt.savefig('./figs/{}'.format(title))
        plt.clf()       
