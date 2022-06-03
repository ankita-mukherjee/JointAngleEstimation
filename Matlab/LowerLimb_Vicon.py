import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

df =  pd.read_csv('dl009_kneeflex_020.csv')
df1 = pd.read_csv('dl009_hipabd_001.csv')  
df2 = pd.read_csv('dl009_hipext_001.csv')  
df3 = pd.read_csv('dl009_hipflex_001.csv')  

# Lengths Knee
hipknee_length = np.sqrt((df['ASI_X_mm'] - df['KNE_X_mm'])**2 + (df['ASI_Z_mm'] - df['KNE_Z_mm'])**2)
hipank_length  = np.sqrt((df['ASI_X_mm'] - df['ANK_X_mm'])**2 + (df['ASI_Z_mm'] - df['ANK_Z_mm'])**2)
kneank_length  = np.sqrt((df['KNE_X_mm'] - df['ANK_X_mm'])**2 + (df['KNE_Z_mm'] - df['ANK_Z_mm'])**2)

# Lengths hipabd
hipknee_length1 = np.sqrt((df1['ASI_X_mm'] - df1['KNE_X_mm'])**2 + (df1['ASI_Z_mm'] - df1['KNE_Z_mm'])**2)
hipank_length1  = np.sqrt((df1['ASI_X_mm'] - df1['ANK_X_mm'])**2 + (df1['ASI_Z_mm'] - df1['ANK_Z_mm'])**2)
kneank_length1  = np.sqrt((df1['KNE_X_mm'] - df1['ANK_X_mm'])**2 + (df1['KNE_Z_mm'] - df1['ANK_Z_mm'])**2)

# Lengths hipext
hipknee_length2 = np.sqrt((df2['ASI_X_mm'] - df2['KNE_X_mm'])**2 + (df2['ASI_Z_mm'] - df2['KNE_Z_mm'])**2)
hipank_length2  = np.sqrt((df2['ASI_X_mm'] - df2['ANK_X_mm'])**2 + (df2['ASI_Z_mm'] - df2['ANK_Z_mm'])**2)
kneank_length2  = np.sqrt((df2['KNE_X_mm'] - df2['ANK_X_mm'])**2 + (df2['KNE_Z_mm'] - df2['ANK_Z_mm'])**2)

# Lengths hipflex
hipknee_length3 = np.sqrt((df3['ASI_X_mm'] - df3['KNE_X_mm'])**2 + (df3['ASI_Z_mm'] - df3['KNE_Z_mm'])**2)
hipank_length3  = np.sqrt((df3['ASI_X_mm'] - df3['ANK_X_mm'])**2 + (df3['ASI_Z_mm'] - df3['ANK_Z_mm'])**2)
kneank_length3  = np.sqrt((df3['KNE_X_mm'] - df3['ANK_X_mm'])**2 + (df3['KNE_Z_mm'] - df3['ANK_Z_mm'])**2)


# Hip Abd
cASIa = (hipknee_length1**2 + hipank_length1**2 - kneank_length1**2) / (2 * hipknee_length1 * hipank_length1)
hipabdangle = np.arccos(cASIa)
thetahipabd = hipabdangle * 180 / math.pi
#thetahip1 = np.real(thetahip)

# Hip Ext
cASIe = (hipknee_length2**2 + hipank_length2**2 - kneank_length2**2) / (2 * hipknee_length2 * hipank_length2)
hipextangle = np.arccos(cASIe)
thetahipext = hipextangle * 180 / math.pi

# Hip Flex
cASIf = (hipknee_length3**2 + hipank_length3**2 - kneank_length3**2) / (2 * hipknee_length3 * hipank_length3)
hipflexangle = np.arccos(cASIf)
thetahipflex = hipflexangle * 180 / math.pi

# Knee
cKNEa = (hipknee_length**2 + kneank_length**2 - hipank_length**2) / (2 * hipknee_length * kneank_length)
kneeangle = np.arccos(cKNEa)
thetaknee = kneeangle * 180 / math.pi


# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(np.concatenate([df['ASI_X_mm'], df['KNE_X_mm']]), np.concatenate([df['ASI_Z_mm'], df['KNE_Z_mm']]), marker="+")
# plt.plot(np.concatenate([df['ASI_X_mm'], df['ANK_X_mm']]), np.concatenate([df['ASI_Z_mm'], df['ANK_Z_mm']]), marker="+")
# plt.plot(np.concatenate([df['KNE_X_mm'], df['ANK_X_mm']]), np.concatenate([df['KNE_Z_mm'], df['ANK_Z_mm']]), marker="+")

# plt.subplot(1, 2, 2)
# plt.plot(np.concatenate([df['ASI_Y_mm'], df['KNE_Y_mm']]), np.concatenate([df['ASI_Z_mm'], df['KNE_Z_mm']]), marker="+")
# plt.plot(np.concatenate([df['ASI_Y_mm'], df['ANK_Y_mm']]), np.concatenate([df['ASI_Z_mm'], df['ANK_Z_mm']]), marker="+")
# plt.plot(np.concatenate([df['KNE_Y_mm'], df['ANK_Y_mm']]), np.concatenate([df['KNE_Z_mm'], df['ANK_Z_mm']]), marker="+")

# plt.savefig('python_fig.png')
# plt.show()

# save in csv
np.savetxt(r'python_dl009_kneeflexoutput_020knee.csv', thetaknee, fmt='%.15f')
np.savetxt(r'python_dl009_hipabdoutput_001hip.csv', thetahipabd, fmt='%.15f')
np.savetxt(r'python_dl009_hipextoutput_001hip.csv', thetahipext, fmt='%.15f')
np.savetxt(r'python_dl009_hipflexoutput_001hip.csv', thetahipflex, fmt='%.15f')