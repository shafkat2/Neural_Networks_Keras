


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
         self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
         self.b += torch.sum((v0-vk),0)
         self.a += torch.sum((ph0 - phk),0)

nv = len(training_set[0])
nh = 100

batch_size = 100

rmb = RBM(nv,nh)

#training the RMB

nb_epoch = 10

for epoch in range(1,nb_epoch + 1):
    train_loss = 0
    s = 0.0
    for id_user in range(0,nb_user - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rmb.sample_h(v0)
        for k in range(10):
            _,hk = rmb.sample_h(vk)
            _,vk = rmb.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rmb.sample_h(vk)
        rmb.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.0
    print("number of epoch: "+str(epoch)+" loss: "+str(train_loss/s) )


#testing the RBM



test_loss = 0
s = 0.0
for id_user in range(0,nb_user):
     v = training_set_toarray[id_user:id_user+1]
     vt = test_set[id_user:id_user+1]
     if( len(vt[vt >=0]) > 0):
         _,h = rmb.sample_h(v)
         _,v = rmb.sample_v(h)
         test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
         s += 1.0
         
print("loss: "+str(test_loss/s) )


