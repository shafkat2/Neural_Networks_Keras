

#importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd


#data importing
dataset  = pd.read_csv("Data_fetching and templates/Data/Credit_Card_Applications.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(x)

X.shape

# training Som
from minisom import MiniSom

som = MiniSom(x = 25, y = 25 , input_len= 15,sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X,num_iteration = 100)

#visualizing
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,c in enumerate(X):
    w = som.winner(c)
    plot(w[0]+0.5,
    w[1]+0.5,
    markers[y[i]],
    markeredgecolor = colors[y[i]],
    markerfacecolor ='None',
    markersize = 10,
    markeredgewidth = 2)

show()
#finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis = 0)
frauds = sc.inverse_transform(frauds)