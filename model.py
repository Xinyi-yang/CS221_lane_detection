import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

%matplotlib inline

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0, rows*1.0]
    top_left     = [cols*0, rows*0.3]
    bottom_right = [cols*1.0, rows*1.0]
    top_right    = [cols*1.0, rows*0.3] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def selectWhiteAndYellow(img):
    """
    selec the white and yellow component in the hsv space.
    (1) set the yellow/white lower and upper bound
    (2) apply the mask to the hsv space image
    """
    lower_yellow = np.array([65, 100, 100], np.uint8)
    upper_yellow = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 20, 255])  # range for H is 0:179
    white_mask = cv2.inRange(img, lower_white, upper_white)

    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(yellow_mask, white_mask))
    return img

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Input: image and parameters
# Output: list of list that contains the start points (x1, y1) and end points (x2, y2)
# of hough lines (line interval)
# Use hough line transformation
def find_slopes(img, minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(img,1,np.pi/180,100, minLineLength, maxLineGap)
    THRES_ANGLE = 10  # if the line angle is between -10 to 10 degrees, lines are discarded
    THRES_SLOPE = math.tan(THRES_ANGLE / 180 * math.pi)
    slopes = []
    intercepts = []
    line_cols = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 != x1:
                slope = float(y2 - y1) / float(x2 - x1)
                if abs(slope) > THRES_SLOPE:
                    slopes.append(slope)
                    inter = y1 - slope * x1
                    intercepts.append(inter)
                    line_cols.append((slope, inter, x1, y1, x2, y2))
    return line_cols


"""
Clustering hough lines
use Silhouette score to find the optimal number of lanes
Output:
num_cluster
final_labels: centers of clusters
clusters: dictionary. key = cluster label, value = lines that belong to this cluster
"""
def find_num_lanes(line_cols):
    # line_cols = (slope, inter, x1, y1, x2, y2)
    MAX_CLUSTER = 5
    MIN_CLUSTER = 2
    line_cols.sort(key=lambda tup: tup[0])
    cluster_slope = 1
    cluster_slope_start = []
    cluster_slope_start.append(0)
    cluster_slope_end = []
    for i in range(len(line_cols)-1):
        if line_cols[i+1][0] - line_cols[i][0] > 0.2:
            cluster_slope +=1
            cluster_slope_end.append(i)
            cluster_slope_start.append(i+1)
    cluster_slope_end.append(n-1)
    # remove clusters where there is only one line (possibly outliers)
    abnormal = []
    to_delete =[]
    for i in range(len(cluster_slope_start)):
        if cluster_slope_end[i] - cluster_slope_start[i] == 0:
            abnormal.append(cluster_slope_end[i])
    print ('total num of lines: ', len(line_cols))
    if len(abnormal) > 0:
        for i in abnormal:
            print ('abnormal:', i)
            to_delete.append(i)
        line_cols = [line_cols[i] for i in range(len(line_cols)) if i not in to_delete]
        cluster_slope -= len(abnormal)
    # constrin the number of clusters between 2 and 5
    num_cluster = max(min(MAX_CLUSTER, cluster_slope), MIN_CLUSTER)
    """
    #x_list = []
    #midpoint = []
    #x_list.append(x1)
    #x_list.append(x2)
    #x1, y1, x2, y2 = zip(*line_cols)
    #x_mid = [(a + b)/2 for a, b in zip(x1, x2)]
    #y_mid = [(a + b)/2 for a, b in zip(y1, y2)]
    #X = np.column_stack((x_mid, y_mid))
    #num_lane_candidate = []
    #for X in x_list:
    slopes = sorted(slopes)
    slopes = np.asarray(slopes)
    candidate = {}
    X = slopes.reshape(-1, 1)
    for n_clusters in CLUSTER_RANGE:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=50).fit(X)
        cluster_labels = clusterer.predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels, metric='euclidean')
        candidate[n_clusters] = silhouette_avg
    print (candidate)
    temp = max(candidate, key=candidate.get)
    print (temp)
    #num_lane_candidate.append(temp)
    num_lane = temp
    
    x1 = np.asarray(slopes)
    x2 = np.asarray(intercepts)
    X = np.column_stack((x1, x2))
    """
    slope_list = [i[0] for i in line_cols]
    X = np.asarray(slope_list)
    X = X.reshape(-1, 1)
    estimator = KMeans(n_clusters=num_cluster).fit(X)
    final_labels = estimator.cluster_centers_
    clusters = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
    return num_cluster, final_labels, clusters


 def findTwoPoints(slope, inter):
    """
    In order to get two stable lanes,

    :param slope: current slope from the regressor for current frame
    :param inter: current intercept from the regressor for current frame
    :return tow points locations, which are the two ends of a lane
    """
    
    # fix the y coordinates of the top and bottom points
    top_y = 250
    bottom_y = 719

    p1_y = bottom_y
    p1_x = int((float(p1_y)-inter)/slope)
    p2_y = top_y
    p2_x = int((float(p2_y)-inter)/slope)

    return (p1_x, p1_y, p2_x, p2_y)


 def regress_a_lane(img, line_cols, clusters, outputrange, color=[255, 0, 0], thickness=10):
    """ regress a line from a cluster of points and add it to img
    (1) use a linear regressor to fit the data (x,y)
    (2) remove outlier, and then fit the cleaned data again to get slope and intercept
    (3) find the two ends of the desired line by using slope and intercept
    
    :param img: input image
    :param line_cols: hough line transformation output
    :param clusters: clustering hough lines into 2-5 groups (indicating 2-5 lines)
    :param color: line color
    :param thickness: thickness of the line  
    """
    # find the two end points of the line by using slope and iter, and then visulize the line
    
    lr = LinearRegression()
    height = img.shape[0]
    lane_pred = []
    if len(clusters) == 0:
        return img
    if len(clusters) == 1:
        ind = clusters[0]
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        for j in ind:
            sl, inter, x1, y1, x2, y2 = line_cols[j]
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        x = x1_list + x2_list
        y = y1_list + y2_list
        x = np.asarray(x)
        y = np.asarray(y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        lr.fit(x, y)
        slope = lr.coef_
        intercept = lr.intercept_
        p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, intercept)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)
        x_labels = [int((float(p_y)-inter)/slope) for p_y in outputrange]
        
    else:
        for i in range(len(clusters)):
            ind = clusters[i]
            x1_list = []
            x2_list = []
            y1_list = []
            y2_list = []
            for j in ind:
                sl, inter, x1, y1, x2, y2 = line_cols[j]
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
            x = x1_list + x2_list
            y = y1_list + y2_list
            x = np.asarray(x)
            y = np.asarray(y)
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            lr.fit(x, y)
            slope = lr.coef_
            intercept = lr.intercept_
            p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, intercept)
            cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)
            for p_y in outputrange:
                p_x = int((float(p_y)-inter)/slope) 
                if p_x > 1280 or p_x < 1:
                    p_x = -2
                onepoint = []
                onepoint.append(i+1)
                onepoint.append(p_x)
                onepoint.append(p_y)
                lane_pred.append(onepoint)
            
    return img, lane_pred
    
"""
    # identify and remove outliers
    cleaned_data = []
    try:
        predictions = reg.predict(x)
        cleaned_data = outlierCleaner(predictions, x, y)
    except NameError:
        print("err in regression prediction")

    if len(cleaned_data) > 0:
        x, y = cleaned_data   
        # refit cleaned data!
        try:
            reg.fit(x, y)
        except NameError:
            print("err in reg.fit for cleaned data")
    else:
        print("outlierCleaner() is returning an empty list, no refitting to be done")
"""


if __name__ == "__main__":
	minLineLength = 150
	maxLineGap = 10

	filename_bw = 'benchmark/BW_v3/20.jpg'
	filename_color = 'benchmark/20.jpg'

	raw = cv2.imread(filename_color)
	raw_bw = cv2.imread(filename_bw)
	img = cv2.imread(filename_bw)
	img_col = cv2.imread(filename_color)
	sobel_display = cv2.imread(filename_color)
	img = select_region(img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength,maxLineGap)

	img_col = select_region(img_col)
	gray_col = cv2.cvtColor(img_col,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray_col,20,150,apertureSize = 3)
	lines_col = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

	n = 0
	THRES_ANGLE = 10  # if the line angle is between -5 to 5 degrees, lines are discarded
	THRES_SLOPE = math.tan(THRES_ANGLE / 180 * math.pi)
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        if x2 != x1:
	            slope = float(y2 - y1) / float(x2 - x1)
	            if abs(slope) > THRES_SLOPE:
	                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	                n += 1

	for line in lines_col:
	    for x1,y1,x2,y2 in line:
	        if x2 != x1:
	            slope = float(y2 - y1) / float(x2 - x1)
	            if abs(slope) > THRES_SLOPE:
	                cv2.line(img_col,(x1,y1),(x2,y2),(0,255,0),2)
	num_lane, final_labels, clusters = find_num_lanes(line_cols)

	outputrange = range(160, 720, 10)
	output_img, lane_pred = regress_a_lane(sobel_display, line_cols, clusters, outputrange)



	#cv2.imwrite('test_temp.jpg',img)
	f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10))
	ax1.imshow(raw, cmap="gray")
	ax1.set_title('Raw image', fontsize=30)
	ax2.imshow(raw_bw, cmap="gray")
	ax2.set_title('Sobel + Binarization Processed Image', fontsize=15)
	ax3.imshow(img_col)
	ax3.set_title('Hough Lines + Canny Edge detection', fontsize=15)
	ax4.imshow(img)
	ax4.set_title('Hough Lines + Sobel Edge detection', fontsize=15)
	ax5.imshow(output_img)
	ax5.set_title('Hough Lines + Sobel Edge detection', fontsize=15)