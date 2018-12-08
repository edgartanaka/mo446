import numpy as np
import cv2 as cv
from sklearn.neighbors import NearestNeighbors

def explore_match(img1, img2, kp1, kp2, file_name):
    """
    Displays the matched keypoints.
    Inspired from https://stackoverflow.com/questions/48220817/how-to-match-and-align-two-images-using-surf-features-python-opencv
    
    :param img1: an array representing an (row,col,chan) image
    :param img2: an array representing an (row,col,chan) image
    :param kp1: an array representing the keypoints for img1
    :param kp2: an array representing the keypoints for img2
    :param file_name: path to the image that will be generated showing the matched keypoints
    """
    
    kp_pairs = zip(kp1, kp2)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2

    kp_pairs = list(kp_pairs)

    p1 = np.int32([kpp[0] for kpp in kp_pairs])
    p2 = np.int32([kpp[1] for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    
    r = 2
    thickness = 3
    
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv.circle(vis, (x1, y1), 2, green, -1)
        cv.circle(vis, (x2, y2), 2, green, -1)
        cv.line(vis, (x1, y1), (x2, y2), green)
    
    cv.imwrite(file_name, vis)
    
    

def matching(kp1, kp2, des1, des2, img1=None, img2=None, file_name=None, r=200):
    """
    Displays the matched keypoints
    :param kp1: an array representing the keypoints for img1
    :param kp2: an array representing the keypoints for img2
    :param des1: an array representing the feature descriptors for img1
    :param des2: an array representing the feature descriptors for img2
    :param img1: an array representing an (row,col,chan) image
    :param img2: an array representing an (row,col,chan) image
    :param file_name: path to the image that will be generated showing the matched keypoints
    :param r: radius to use in the neighbors search
    """
       
    neigh1 = NearestNeighbors(1, algorithm="kd_tree")
    neigh1.fit(des1)
    
    nearest_dist1, nearest_ind1 = neigh1.kneighbors(des2, 1)
    indexes_12 = nearest_dist1 < r
    indexes_11 = nearest_ind1[indexes_12]

    if img1 is not None and img2 is not None:
        explore_match(img1, img2, kp1[indexes_11], kp2[indexes_12.flatten()], file_name)
    
    return kp1[indexes_11], kp2[indexes_12.flatten()]
    

if __name__ == "__main__":
    from glob import glob
    
    def SIFT(img_path):
        img = cv.imread(img_path)
        working_image = img.copy()
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        img_draw = cv.drawKeypoints(img,kp,img)

        return working_image, kp, des
    
    
    for folder in ['rotate','scale','translate']:
        files = glob(f'../input/frames/{folder}/*.jpg')
        
        for i in range(len(files)-1):
            img1, kp1, des1 = SIFT(f'../input/frames/{folder}/frame{i}.jpg')
            img2, kp2, des2 = SIFT(f'../input/frames/{folder}/frame{i+1}.jpg')

            kp1 = np.array([i.pt for i in kp1])
            kp2 = np.array([i.pt for i in kp2])

            matching(kp1, kp2, des1, des2, img1, img2, f"../output/feature_matching_{i}.jpg",120)
