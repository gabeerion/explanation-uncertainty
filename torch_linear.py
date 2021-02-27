import numpy as np
import torch


class TorchLinear(object):
    def __init__(self,lr=0.01,max_iter=1000):
        self.lr=lr
        self.max_iter=max_iter
        
    def _get_model(self,d):
        self.linear_layer = torch.nn.Linear(in_features=d,out_features=1,bias=True)
        return torch.nn.Sequential(
            self.linear_layer
        )
    
    def _fit_internal(self,model,opt,Xtensor,ytensor):
        self.train_scores=[]
        for i in range(self.max_iter):
            opt.zero_grad()
            preds = model(Xtensor)
            mse = torch.nn.functional.mse_loss(preds.flatten(),ytensor)
            self.train_scores.append(mse.item())
            mse.backward()
            opt.step()
    
    def fit(self,X,y):
        Xtorch = torch.Tensor(X)
        ytorch = torch.Tensor(y)
        
        self.model = self._get_model(X.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        
        self._fit_internal(self.model,self.optimizer,Xtorch,ytorch)
        
        self.coef_ = self.linear_layer.weight.detach().numpy().flatten()
        self.intercept_ = self.linear_layer.bias.detach().item()
        
        return self
    
    def predict(self,X):
        Xtorch = torch.Tensor(X)
        return self.model(Xtorch).detach().numpy().flatten()