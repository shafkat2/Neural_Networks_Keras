
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