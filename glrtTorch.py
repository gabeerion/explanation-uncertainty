# Estimate confidence intervals on the coefficients using glrt likelihood bounds + attribution regularization

import torch
import numpy as np
from glrt_stat import bootstrapGLRTcis
from attributionpriors.pytorch_ops import ExpectedGradientsModel

def lowCoefObj(llh,coef,lmbd):
    return llh+lmbd*coef
def highCoefObj(llh,coef,lmbd):
    return llh-lmbd*coef

def glrtTorchCis(modelFn,X,y,alpha=0.05,bootstrap_kwargs={}):
    lcb_LR, ucb_LR = bootstrapGLRTcis(modelFn, X, y, alpha=0.05, **bootstrap_kwargs)
    lcbs, ucbs = [], []
    for idx in range(X.shape[1]):
        ucb_Coef = getBoundary(modelFn,X,y,idx,obj=highCoefObj)
        lcb_Coef = getBoundary(modelFn,X,y,idx,obj=lowCoefObj)
        lcbs.append(lcb_Coef)
        ucbs.append(ucb_Coef)
    return np.array(lcbs), np.array(ucbs)

def getBoundary(modelFn,X,y,idx,obj):
    Xtorch = torch.Tensor(X)
    ytorch = torch.Tensor(y)
    Rtorch = torch.ones_like(X)*X.mean(0).reshape(1,-1)
    WrapperModel = modelFn()
    TorchModel = WrapperModel._init_model()
    IGModel = ExpectedGradientsModel(TorchModel,Rtorch,k=10,random_alpha=False,scale_by_inputs=True)
    
def trainWithAttributions(model,X,y,obj):
        self.train_scores=[]
        for i in range(self.max_iter):
            opt.zero_grad()
            preds = model(Xtensor)
            mse = torch.nn.functional.mse_loss(preds.flatten(),ytensor)
            self.train_scores.append(mse.item())
            mse.backward()
            opt.step()