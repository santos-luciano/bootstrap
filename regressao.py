import statsmodels.api as sm
import pandas as pd

bra_olym = pd.read_csv("bra_olym.csv")

def GLM(data,modelo):
  y = data.iloc[:,:1]
  X = data.iloc[:,1:]
  X = sm.add_constant(X)
  if modelo == 'gaussian':
    model = sm.GLM(y,X,family=sm.families.Gaussian())
  elif modelo == 'poisson':
    model = sm.GLM(y,X,family=sm.families.Poisson())
  elif modelo == 'binomialnegativa':
    model = sm.GLM(y,X,family=sm.families.NegativeBinomial())
  regressao = model.fit()
  return pd.DataFrame(regressao.params).T

    