from subprocess import call
import re
import sys
import os
import numpy as np
file_name = sys.argv[1]
fp = open(file_name)
contents = fp.read()
k = np.zeros((3,3))
imgL, imgR, k[0,0], k[0,1], k[0,2], k[1,0], k[1,1], k[1,2], k[2,0], k[2,1], k[2,2] = contents.split()
os.chdir("build")
call(["cmake", "."])
call(["make"])
os.chdir("..")
call(["python3", "src/main.py", file_name])
call(["./build/3D_Reconstruction", imgL, "generated/disparity-image.pgm", "generated/Q.xml"])
