import scikits.bootstrap as IntervaloBoot

def IntervalosBCA(data):
  li_b0,ls_b0 = IntervaloBoot.ci(data.iloc[:,0])
  li_b1,ls_b1 = IntervaloBoot.ci(data.iloc[:,1])
  li_b2,ls_b2 = IntervaloBoot.ci(data.iloc[:,2])
  return li_b0,ls_b0,li_b1,ls_b1,li_b2,ls_b2