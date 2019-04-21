
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable  

#importing the dataset
movies = pd.read_csv('Data_fetching and templates/Data/ml-1m/movies.dat', sep = '::',header = None,engine = 'python',encoding = 'latin-1')
users = pd.read_csv('Data_fetching and templates/Data/ml-1m/users.dat', sep = '::',header = None,engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv('Data_fetching and templates/Data/ml-1m/ratings.dat', sep = '::',header = None,engine = 'python',encoding = 'latin-1')

#preapring Training set and test set
training_set = pd.read_csv('Data_fetching and templates/Data/ml-100k/u1.base',delimiter= '\t' )
training_set_toarray = np.array(training_set, dtype= 'int')

test_set = pd.read_csv('Data_fetching and templates/Data/ml-100k/u1.test',delimiter= '\t' )
test_set_toarray = np.array(test_set, dtype= 'int')

#getting the number of the users and movies
nb_user = int(max(max(training_set_toarray[:,0]),max(test_set_toarray[:,0])))
nb_movies = int(max(max(training_set_toarray[:,1]),max(test_set_toarray[:,1])))
#converting the data into array with users in line movies in colmuns
def convert(data):
    new_data = []
    for id_users in range(1,nb_user+1):
        id_movies = data[:,1][data[:,0] == id_users ]
        id_ratings = data[:,2][data[:,0] == id_users ]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set_to_list  = convert(training_set_toarray )
test_set_to_list  = convert(test_set_toarray )

#coverting data into tensor
training_set = torch.Tensor(training_set_to_list)
test_set = torch.FloatTensor(test_set_to_list)

#rating into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = - 1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1


test_set[test_set == 0] = - 1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#creating architecture of the neural network

class RBM():
    def __init__(self,nv,nh):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation  = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return  p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation  = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return  p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,v0, vk, ph0, phk):
         self.W += torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk) 
         self.b += torch.sum((v0-vk),0)
         self.a += torch.sum((ph0 - phk),0)
