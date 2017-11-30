import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def calcProjectionMat(k, R, T):
    return np.matmul(k,np.column_stack((R,T)))

def findEssentialFromFundamental(k,F):
    return np.matmul(k.T, np.matmul(F, k))

def findKeyMatchesAndDescriptors(imgLeft, imgRight):
    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgLeft,None)
    kp2, des2 = sift.detectAndCompute(imgRight,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def findFundamentalMatrix(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    #pts1 = pts1[mask.ravel()==1]
    #pts2 = pts2[mask.ravel()==1]
    return F

def findProjectionsFromEssentialandFundamental(E):
    U, s, V = np.linalg.svd(E, full_matrices=True)
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R = np.matmul(U,np.matmul(W.T,V.T))
    T = U[:,2:3]*-1
    return R, T

def write_Q_to_file(Q):
    Q=np.ndarray.flatten(Q)
    file = open("generated/Q.xml",'w')
    file.write("<?xml version=\"1.0\"?>\n")
    file.write("<opencv_storage>\n")
    file.write("<Q type_id=\"opencv-matrix\">\n")
    file.write("<rows>4</rows>\n")
    file.write("<cols>4</cols>\n")
    file.write("<dt>d</dt>\n")
    file.write("<data>\n")
    file.write(str(Q[0])+" "+str(Q[1])+" "+str(Q[2])+" "+str(Q[3])+" "+str(Q[4])+" "+str(Q[5])+" "+str(Q[6])+" "+str(Q[7])+" "+str(Q[8])+" "+str(Q[9])+" "+str(Q[10])+" "+str(Q[11])+" "+str(Q[12])+" "+str(Q[13])+" "+str(Q[14])+" "+str(Q[15]))
    file.write("</data></Q>\n")
    file.write("</opencv_storage>\n")
    file.close()

def drawlines(img1,img2,lines,pts1,pts2):
    # img1 - image on which we draw the epilines for the points in img2
    # lines - corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def compute_visualize_epilines(imgLeft, imgRight, F, pts1, pts2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(imgLeft,imgRight,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(imgRight,imgLeft,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


# read images and camera matrix
file_name = sys.argv[1]
fp = open(file_name)
contents = fp.read()
k = np.zeros((3,3))
imgL, imgR, k[0,0], k[0,1], k[0,2], k[1,0], k[1,1], k[1,2], k[2,0], k[2,1], k[2,2] = contents.split()
imgLeft = cv2.imread(imgL,0)
imgRight = cv2.imread(imgR,0)

# calculate and save disparity to generated dir
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)
disparity = stereo.compute(imgLeft,imgRight)
cv2.imwrite('generated/disparity-image.pgm', disparity)

# find key mathces and fundamental matrix
pts1, pts2 = findKeyMatchesAndDescriptors(imgLeft, imgRight)
F = findFundamentalMatrix(pts1, pts2)
#compute_visualize_epilines(imgLeft, imgRight, F, pts1, pts2)

# find essential from fundamentl matrix
E = findEssentialFromFundamental(k, F)

# find R and T upto projective ambiguity
R, T = findProjectionsFromEssentialandFundamental(E)

# assume no distortion
d=np.array([0,0,0,0,0])

# calculate Q i.e. disparity to depth conversion matrix from R and T and write it to a file
R_rectified1, R_rectified2, P_rectified1, P_rectified2, Q, roi1, roi2 = cv2.stereoRectify(k, d,
                                                      k, d,
                                                      imgLeft.shape[:2],
                                                      R, T, alpha=1)
print("Q Matrix:\n",Q)
write_Q_to_file(Q)

# use cpp program to visualize in point cloud since PCL visualization is not supported in brew python
# AND Martin Peris has already done that so we can reuse it
##########################################END#########################################################


#Image3D = cv2.reprojectImageTo3D(disparity, Q, ddepth=cv2.CV_32F)
#print(Image3D)
#Ys = Image3D[:, 0]
#Zs = Image3D[:, 1]
#Xs = Image3D[:, 2]
#R1 = np.array([[0.02187598221295043000, 0.98329680886213122000, -0.18068986436368856000], [0.99856708067455469000, -0.01266114646423925600, 0.05199500709979997700], [0.04883878372068499500, -0.18156839221560722000, -0.98216479887691122000]])
#T1 = np.array([-0.0726637729648, 0.0223360353405, 0.614604845959])
#R2 = np.array([[-0.03472199972816788400, 0.98429285136236500000, -0.17309524976677537000], [0.93942192751145170000, -0.02695166652093134900, -0.34170169707277304000], [-0.34099974317519038000, -0.17447403941185566000, -0.92373047190496216000]])
#T2 = np.array([-0.0746307029819, 0.0338148092011, 0.600850565131])
#P1 = calcProjectionMat(k, R1, T1)
#P2 = calcProjectionMat(k, R2, T2)
#points4D = cv2.triangulatePoints(P1, P2, pts1.T.astype(float), pts2.T.astype(float))
#print(points4D)
#P1_estimated = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#P2_estimated = calcProjectionMat(np.identity(3), R, T)
#points4D = cv2.triangulatePoints(P1_estimated, P2_estimated, pts1.T.astype(float), pts2.T.astype(float))
#points4D_new = np.reshape(points4D,(points4D.shape[0],1,points4D.shape[1]))
#print(points4D)
#points3D = np.zeros((3,points4D.shape[1]))
#points3D[0,:] = points4D[0,:]/points4D[3,:]
#points3D[1,:] = points4D[1,:]/points4D[3,:]
#points3D[2,:] = points4D[2,:]/points4D[3,:]
#print(points3D)
