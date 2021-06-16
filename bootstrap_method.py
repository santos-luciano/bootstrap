import numpy as np


def Bootstrap(data,R,modelo):
  betas = pd.DataFrame()
  for i in range(R):
    indices = np.random.choice(range(len(data)),len(data))
    aux1 = pd.DataFrame()
    for j in indices:
      aux0 = data.loc[[j]]
      aux1 = pd.concat([aux1,aux0])
    betas = pd.concat([betas,GLM(aux1,modelo)])
  betas.index = range(len(betas))
  return betas

def BootstrapBayesiano(data,R,modelo):
  betas = pd.DataFrame()
  for i in range(R): 
    vetor_aux = [0]
    vetor_aux.extend(np.random.uniform(0,1,len(data)-1))    
    vetor_aux.extend([1])
    vetor_aux = sorted(vetor_aux)
    vetor_prop = []
    for v in range(1,len(vetor_aux)):
      vetor_prop.append(vetor_aux[v]-vetor_aux[v-1])
    indices = np.random.choice(range(len(data)),len(data),p = vetor_prop)
    aux1 = pd.DataFrame()
    for j in indices:
      aux0 = data.loc[[j]]
      aux1 = pd.concat([aux1,aux0])
    betas = pd.concat([betas,GLM(aux1,modelo)])
  betas.index = range(len(betas))
  return betas

def BootstrapSuavizado(data,R,modelo):
  betas = pd.DataFrame()
  for i in range(R):
    indices = np.random.choice(range(len(data)),len(data))
    aux1 = pd.DataFrame()
    for j in indices:
      aux0 = data.loc[[j]]
      aux1 = pd.concat([aux1,aux0])
    s_2 = (1/(len(aux1)-1))*((aux1 - aux1.mean())**2).sum()
    s = s_2**(1/2)
    h = s/(len(aux1)**2)
    erro1 = np.random.normal(loc = 0,scale = h[0]**2,size = len(aux1))
    erro2 = np.random.normal(loc = 0,scale = h[1]**2,size = len(aux1))
    erro3 = np.random.normal(loc = 0,scale = h[2]**2,size = len(aux1))
    aux1.iloc[:,0] = aux1.iloc[:,0]+erro1
    aux1.iloc[:,1] = aux1.iloc[:,1]+erro2
    aux1.iloc[:,2] = aux1.iloc[:,2]+erro3
    betas = pd.concat([betas,GLM(aux1,modelo)])
  betas.index = range(len(betas))
  return betas

def BootstrapDuplo(data,R,R1,modelo):
  betas = pd.DataFrame()
  for i in range(R):
    indices = np.random.choice(range(len(data)),len(data))
    aux1 = pd.DataFrame()
    for j in indices:
      aux0 = data.loc[[j]]
      aux1 = pd.concat([aux1,aux0])
    aux1.index = range(len(indices))
    betas0 = pd.DataFrame()
    for l in range(R1):
      indices1 = np.random.choice(range(len(aux1)),len(aux1))
      aux3 = pd.DataFrame()
      for m in indices1:
        aux2 = aux1.loc[[m]]
        aux3 = pd.concat([aux3,aux2])
      betas0 = pd.concat([betas0,GLM(aux3,modelo)])
    betas = pd.concat([betas,pd.DataFrame(betas0.mean()).T])
  betas.index = range(len(betas))
  return betas

def BootstrapDuploRapido(data,R,modelo):
  betas = pd.DataFrame()
  for i in range(R):
    indices = np.random.choice(range(len(data)),len(data))
    aux1 = pd.DataFrame()
    for j in indices:
      aux0 = data.loc[[j]]
      aux1 = pd.concat([aux1,aux0])
    indices1 = np.random.choice(range(len(aux1)),len(aux1))
    aux3 = pd.DataFrame()
    aux1.index = range(len(aux1))
    for l in indices1:
      aux2 = aux1.loc[[l]]
      aux3 = pd.concat([aux3,aux2])
    betas = pd.concat([betas,GLM(aux3,modelo)])
  betas.index = range(len(betas))
  return betas

def BootstrapBootknife(data,R,modelo):
  n = len(data)
  k = int(R/n)
  betas = pd.DataFrame()
  for i in range(n):
    data1 = data.drop(i)
    betas0 = pd.DataFrame()
    for j in range(k):
      data1.index = range(len(data1))
      indices = np.random.choice(range(len(data1)),len(data1))
      aux1 = pd.DataFrame()
      for l in indices:
        aux0 = data1.loc[[l]]
        aux1 = pd.concat([aux1,aux0])
      betas0 = pd.concat([betas,GLM(aux1,modelo)])
    betas = pd.concat([betas,pd.DataFrame(betas0.mean()).T])
  betas.index = range(len(betas))
  return betas
