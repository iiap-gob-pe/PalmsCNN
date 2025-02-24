from scipy.io import savemat
import numpy as np
a = np.arange(20)
mdic = {"a": a, "label": "experiment"}
print(mdic)
savemat("D:\\VICERCAVI\\IIAP\\HPC\\Trabajos_Conjuntos\\xtagle\\dwt\\model\\direction_unified_CR_unified_CR_pretrain_005.mat", mdic)
print("save")