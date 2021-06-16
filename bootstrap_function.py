def BootstrapGeral(data,metodo,modelo,R,R1=None):
  if metodo == 'classico':
    boot = Bootstrap(data,R,modelo)
  elif metodo == 'bayesiano':
    boot = BootstrapBayesiano(data,R,modelo)
  elif metodo == 'duplo':
    boot = BootstrapDuplo(data,R,R1,modelo) 
  elif metodo == 'duplorapido':
    boot = BootstrapDuploRapido(data,R,modelo)
  elif metodo == 'bootknife':
    boot = BootstrapBootknife(data,R,modelo)
  elif metodo == 'suavizado':
    boot = BootstrapSuavizado(data,R,modelo)
  parametros = GLM(data,modelo)
  b0_boot,b1_boot,b2_boot = boot.mean()
  b0_li,b0_ls,b1_li,b1_ls,b2_li,b2_ls = IntervalosBCA(boot)
  errob0,errob1,errob2 = [boot.iloc[:,0].std(),boot.iloc[:,1].std(),boot.iloc[:,2].std()]
  viesb0,viesb1,viesb2 = [boot.iloc[:,0].mean()-parametros.iloc[0,0],boot.iloc[:,1].mean()-parametros.iloc[0,1],boot.iloc[:,2].mean()-parametros.iloc[0,2]]
  vies = [viesb0,viesb1,viesb2]
  erro = [errob0,errob1,errob2]
  li = [b0_li,b1_li,b2_li]
  ls = [b0_ls,b1_ls,b2_ls]
  estimativa_corrigida = [2*parametros.iloc[0,0]-boot.iloc[:,0].mean(),2*parametros.iloc[0,1]-boot.iloc[:,1].mean(),2*parametros.iloc[0,2]-boot.iloc[:,2].mean()]
  teste = pd.DataFrame({'boot':boot.mean(),'li':li,'ls':ls,'erro':erro,'vies':vies})
  teste.index = ['b0','b1','b2']
  teste1 = pd.DataFrame({'estimativa original':parametros.iloc[0],'estimativa corrigida':estimativa_corrigida})
  geral = {'boot':teste,'corrigida':teste1}
  return geral

BootstrapGeral(bra_olym,'classico','gaussian',10)