import numpy as np
import os


landmarks = np.load(os.path.join("./dataset/WFLW_test/landmarks","0_Parade_marchingband_1_702_377.pts.npy"))
print(landmarks.dtype)