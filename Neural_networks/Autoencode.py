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
training_set = torch.FloatTensor(training_set_to_list)
test_set = torch.FloatTensor(test_set_to_list)

#creating the architecures of the neural netowrk
class SAE(nn.Module):
    def __init__(self):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(nb_movies,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay = 0.5)

#training the SAE

nb_epoch = 200

for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.0
    for id_user in range(nb_user):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) >0:
            output = sae(input)