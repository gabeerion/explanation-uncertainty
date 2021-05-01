# Estimate confidence intervals on the coefficients using glrt likelihood bounds + attribution regularization

from functools import partial
import torch
import numpy as np
from .glrt_stat import bootstrapGLRTcis
from attributionpriors.pytorch_ops import ExpectedGradientsModel

def lowCoefObj(idx,lmbd,llh,global_attributions):
    """
    Objective to minimize loss and *minimize* attribution to the feature at 'idx'.
    :param idx: Feature index whose attribution should be minimized
    :param lmbd: Strength of attribution penalization
    :param llh: Loss; positive, where lower is a better value
    :param global_attributions: Vector of attributions for each feature
    
    :return: A scalar total loss
    """
    return llh+lmbd*global_attributions[idx]
def highCoefObj(idx,lmbd,llh,global_attributions):
    """
    Objective to minimize loss and *maximize* attribution to the feature at 'idx'.
    :param idx: Feature index whose attribution should be maximized
    :param lmbd: Strength of attribution penalization
    :param llh: Loss; positive, where lower is a better value
    :param global_attributions: Vector of attributions for each feature
    
    :return: A scalar total loss
    """
    return llh-lmbd*global_attributions[idx]

def MSE(y, yPred):
    """
    Mean Squared Error objective
    :param y: True labels
    :param yPred: Predictions
    
    :return: A scalar MSE loss
    """
    return np.mean((y - yPred)**2)

def glrtTorchCis(modelFn,X,y,alpha=0.05,citype='attribs',bootstrap_kwargs={},search_kwargs={},fit_kwargs={}):
    """
    High level function to give confidence intervals on model attributions
    :param modelFn: Function with no arguments; returns a model
    :param X: Covariates
    :param y: Labels
    :param alpha: Probability of allowing a type I error
    :param citype: One of 'attribs' or 'coefs': Whether to return CIs on IG attributions or raw linear
        regression coefficients
    :param bootstrap_kwargs: Keyword arguments to the bootstrapGLRTcis bootstrapping function
    :param search_kwargs: Keyword arguments to the getBoundaryCoef search function
    :param fit_kwargs: Keyword arguments to the model fit function
    
    :return: array of lower bounds, array of upper bounds, additional results for debugging
    """
    #lcb_LR, ucb_LR = bootstrapGLRTcis(modelFn, X, y, MSE, alpha=alpha, **bootstrap_kwargs)
    ucb_LR = bootstrapGLRTcis(modelFn, X, y, MSE, alpha=alpha, **bootstrap_kwargs)
    lcbs, ucbs = [], []
    lcb_all_results, ucb_all_results = [], []
    for idx in range(X.shape[1]):
        if citype=='attribs':
            ucb_Coef, ucb_Results = getBoundary(modelFn,X,y,idx,ucb=ucb_LR,obj=highCoefObj,reduction=np.max,fit_kwargs=fit_kwargs,**search_kwargs)
            lcb_Coef, lcb_Results = getBoundary(modelFn,X,y,idx,ucb=ucb_LR,obj=lowCoefObj,reduction=np.min,fit_kwargs=fit_kwargs,**search_kwargs)
        elif citype=='coefs':
            ucb_Coef, ucb_Results = getBoundaryCoef(modelFn,X,y,idx,ucb=ucb_LR,obj=highCoefObj,reduction=np.max,fit_kwargs=fit_kwargs,**search_kwargs)
            lcb_Coef, lcb_Results = getBoundaryCoef(modelFn,X,y,idx,ucb=ucb_LR,obj=lowCoefObj,reduction=np.min,fit_kwargs=fit_kwargs,**search_kwargs)
        else: raise ValueError('Only attribs and coefs are supported citypes!')
        lcbs.append(lcb_Coef)
        ucbs.append(ucb_Coef)
        lcb_all_results.append(lcb_Results)
        ucb_all_results.append(ucb_Results)
    #return np.array(lcbs), np.array(ucbs), lcb_all_results, ucb_all_results, lcb_LR, ucb_LR
    return np.array(lcbs), np.array(ucbs), lcb_all_results, ucb_all_results, ucb_LR

def getBoundary(modelFn,X,y,idx,ucb,obj=lowCoefObj,reduction=np.min,lmbds=np.logspace(-10,10,101),lossfunc=torch.nn.functional.mse_loss,fit_kwargs={}):
    """
    Use attribution priors code to search for boundaries on attributions
    :param modelFn: Function with no arguments; returns a model
    :param X: Covariates
    :param y: Labels
    :param idx: Index of feature to search attributions for
    :param obj: Loss function combining label loss and attribution loss. Existing options lowCoefObj and highCoefObj
        will find lower and upper bounds on feature attribution, respectively
    :param reduction: Function to choose from among "acceptable-quality" attribution values. np.min gives 
        lower bound, and np.max gives upper bound, respectively
    :param lmbds: Attribution prior penalties to search; more and finer grid gives more accurate results
    :param lossfunc: Torch loss taking labels and predictions
    :param fit_kwargs: Keyword arguments to the model fit function
    
    :return: Scalar lower boundary value of attribution, plus tuple of
        (all MSEs, attributions, regression coefs, and biases)
    """
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
        bias = WrapperModel.linear_layer.bias.detach().item() if WrapperModel.linear_layer.bias is not None else 0.0
        mses.append(mse.item())
        attributions.append(attrib.item())
        coefs.append(coef)
        biases.append(bias)
        
    mses, attributions, coefs, biases = np.array(mses), np.array(attributions), np.array(coefs), np.array(biases)
    return reduction(attributions[mses<=ucb]), (mses, attributions, coefs, biases)
    
def trainWithAttributions(model,X,y,obj,lossfunc,lr=0.001,max_iter=1000):
    """
    Train with model, data, objective, and attribution prior penalty
    """
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

def getBoundaryCoef(modelFn,X,y,idx,ucb,obj=lowCoefObj,reduction=np.min,lmbds=np.logspace(-10,10,101),lossfunc=torch.nn.functional.mse_loss,fit_kwargs={}):
    """
    Use attribution priors code to search for boundaries on coefficients
    :param modelFn: Function with no arguments; returns a model
    :param X: Covariates
    :param y: Labels
    :param idx: Index of feature to search attributions for
    :param obj: Loss function combining label loss and attribution loss. Existing options lowCoefObj and highCoefObj
        will find lower and upper bounds on feature attribution, respectively
    :param reduction: Function to choose from among "acceptable-quality" attribution values. np.min gives 
        lower bound, and np.max gives upper bound, respectively
    :param lmbds: Attribution prior penalties to search; more and finer grid gives more accurate results
    :param lossfunc: Torch loss taking labels and predictions
    :param fit_kwargs: Keyword arguments to the model fit function
    
    :return: Scalar lower boundary value of attribution, plus tuple of
        (all MSEs, attributions, regression coefs, and biases)
    """
    Xtorch = torch.Tensor(X)
    ytorch = torch.Tensor(y)
    Rtorch = torch.ones_like(Xtorch)*Xtorch.mean(0).reshape(1,-1)
    mses, coefs, biases, attributions = [], [], [], []
    for lmbd in lmbds:
        WrapperModel = modelFn()
        TorchModel = WrapperModel._init_model(X.shape[1])
        TorchModel.coefs = WrapperModel.linear_layer.weight.flatten()
        idxObj = partial(obj,idx,lmbd)
        trainWithCoefs(TorchModel,Xtorch,ytorch,idxObj,lossfunc,**fit_kwargs)
        preds = TorchModel(Xtorch)
        mse = lossfunc(ytorch,preds.flatten())
        coef = WrapperModel.linear_layer.weight.detach().numpy().flatten()
        bias = WrapperModel.linear_layer.bias.detach().item() if WrapperModel.linear_layer.bias is not None else 0.0
        attrib = coef[idx]
        mses.append(mse.item())
        attributions.append(attrib.item())
        coefs.append(coef)
        biases.append(bias)
        
    mses, attributions, coefs, biases = np.array(mses), np.array(attributions), np.array(coefs), np.array(biases)
    return reduction(attributions[mses<=ucb]), (mses, attributions, coefs, biases)
    
def trainWithCoefs(model,X,y,obj,lossfunc,lr=0.001,max_iter=1000):
    """
    Train with model, data, objective, and penalty on coef values
    """
    train_scores=[]
    opt = torch.optim.SGD(model.parameters(),lr=lr)
    for i in range(max_iter):
        opt.zero_grad()
        preds = model(X)
        mse = lossfunc(preds.flatten(),y)
        global_attribs = model.coefs
        total_obj = obj(mse,global_attribs)

        train_scores.append((total_obj.item(),mse.item()))
        total_obj.backward()
        opt.step()