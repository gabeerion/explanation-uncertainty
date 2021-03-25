# Estimate confidence intervals on the coefficients using glrt likelihood bounds + attribution regularization

from functools import partial
import torch
import numpy as np
from glrt_stat import bootstrapGLRTcis
from attributionpriors.pytorch_ops import ExpectedGradientsModel

def lowCoefObj(idx,lmbd,llh,global_attributions):
    return llh+lmbd*global_attributions[idx]
def highCoefObj(idx,lmbd,llh,global_attributions):
    return llh-lmbd*global_attributions[idx]
def MSE(y, yPred):
    return np.mean((y - yPred)**2)

def glrtTorchCis(modelFn,X,y,alpha=0.05,bootstrap_kwargs={},search_kwargs={},fit_kwargs={}):
    lcb_LR, ucb_LR = bootstrapGLRTcis(modelFn, X, y, MSE, alpha=alpha, **bootstrap_kwargs)
    lcbs, ucbs = [], []
    lcb_all_results, ucb_all_results = [], []
    for idx in range(X.shape[1]):
        ucb_Coef, ucb_Results = getBoundary(modelFn,X,y,idx,ucb=ucb_LR,obj=highCoefObj,reduction=np.max,fit_kwargs=fit_kwargs,**search_kwargs)
        lcb_Coef, lcb_Results = getBoundary(modelFn,X,y,idx,ucb=ucb_LR,obj=lowCoefObj,reduction=np.min,fit_kwargs=fit_kwargs,**search_kwargs)
        lcbs.append(lcb_Coef)
        ucbs.append(ucb_Coef)
        lcb_all_results.append(lcb_Results)
        ucb_all_results.append(ucb_Results)
    return np.array(lcbs), np.array(ucbs), lcb_all_results, ucb_all_results

def getBoundary(modelFn,X,y,idx,ucb,obj=lowCoefObj,reduction=np.min,lmbds=np.logspace(-10,10,101),lossfunc=torch.nn.functional.mse_loss,fit_kwargs={}):
    Xtorch = torch.Tensor(X)
    ytorch = torch.Tensor(y)
    Rtorch = torch.ones_like(Xtorch)*Xtorch.mean(0).reshape(1,-1)
    mses, coefs, biases, attributions = [], [], [], []
    for lmbd in lmbds:
        WrapperModel = modelFn()
        TorchModel = WrapperModel._init_model(X.shape[1])
        IGModel = ExpectedGradientsModel(TorchModel,torch.utils.data.TensorDataset(Rtorch),k=10,random_alpha=False,scale_by_inputs=True)
        idxObj = partial(obj,idx,lmbd)
        trainWithAttributions(IGModel,Xtorch,ytorch,idxObj,lossfunc,**fit_kwargs)
        preds, attribs = IGModel(Xtorch,shap_values=True)
        global_attribs = attribs.abs().mean(0)
        mse = lossfunc(ytorch,preds.flatten())
        attrib = global_attribs.flatten()[idx]
        coef = WrapperModel.linear_layer.weight.detach().numpy().flatten()
        bias = WrapperModel.linear_layer.bias.detach().item()
        mses.append(mse.item())
        attributions.append(attrib.item())
        coefs.append(coef)
        biases.append(bias)
        
    mses, attributions, coefs, biases = np.array(mses), np.array(attributions), np.array(coefs), np.array(biases)
    return reduction(attributions[mses<=ucb]), (mses, attributions, coefs, biases)
    
def trainWithAttributions(model,X,y,obj,lossfunc,lr=0.001,max_iter=1000):
        train_scores=[]
        opt = torch.optim.SGD(model.parameters(),lr=lr)
        for i in range(max_iter):
            opt.zero_grad()
            preds, attribs = model(X,shap_values=True)
            mse = lossfunc(preds.flatten(),y)
            global_attribs = attribs.abs().mean(0)
            total_obj = obj(mse,global_attribs)
            
            train_scores.append((total_obj.item(),mse.item()))
            total_obj.backward()
            opt.step()