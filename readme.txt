Prerequisite:
1) PCL library version >=1.6 and all of its dependencies.Refer http://pointclouds.org/
command: brew install pcl

2) Python3 with opencv module in it
command: 
brew install python3
pip3 install opencv-contrib-python

Note that this program is only tested on Mac High Sierra.

How to use:
1) cd into the directory 3D_Reconstruction

2) Create a text file with values (space separated) left_image_path right_image_path k00 k01 k02 k10 k11 k12 k20 k21 k22 (current example text file is in data dir)
Note that paths can be absolute or can be wrt directory 3D_Reconstrution

3) Then call run.py <path_to_txt_file>

Sources:
Images are taken from http://vision.middlebury.edu/mview/ where they provide images with camera matrix and some code is taken from http://martinperis.blogspot.com/2012/01/3d-reconstruction-with-opencv-and-point.html for point cloud visualization