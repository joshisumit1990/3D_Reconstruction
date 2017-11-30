1) Create a text file with values (space separated) left_image_path right_image_path k00 k01 k02 k10 k11 k12 k20 k21 k22 (current example is in data dir)
Note that paths can be absolute or can be wrt directory 3D_Reconstrution

2) Then call run.py <path_to_txt_file>

Note: Images are taken from http://vision.middlebury.edu/mview/ where they provide images with camera matrix and some code is taken from http://martinperis.blogspot.com/2012/01/3d-reconstruction-with-opencv-and-point.html for point cloud visualization