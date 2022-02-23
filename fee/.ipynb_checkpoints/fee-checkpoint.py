# Packages

import dlib
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import joblib
from scipy.spatial import ConvexHull
from skimage.feature import hog
from matplotlib.offsetbox import (TextArea ,AnnotationBbox)
from skimage.morphology.convex_hull import grid_points_in_poly




######### Functions to call


def faceLandmarks(image, predictor):
    """
    Detects the face and landmarks in an image

    Args: image, [numpy.ndarray]
          predictor, this object is a tool that takes in an image region containing
                     some object and outputs a set of point locations that define
                     the pose of the object the [_dlib_pybind11.shape_predictor]

    Returns: landmarks, an array containing the landmarks (68,2) [numpy.ndarray]
             rects, the coordinates (upperLeftX, upperLeftY, bottomleftX, bottomleftY)
                    of the face [list]
    """



    detector = dlib.get_frontal_face_detector()
    detect = detector(image,1)
    shape = predictor(image,detect[0]) #the landmarks in another form
    landmarks = shape_to_np(shape)
    rects = detector(image, 1)
    for k, d in enumerate(rects):
        x = d.left()
        y = d.top()
        w = d.right()
        h = d.bottom()
    rects = [x,y,w,h]
    return landmarks, rects


def alignFromLandmarks(image, predictor):
    """
    Aligns the image given from the landmarks

    Args: image, [numpy.ndarray]
          predictor, this object is a tool that takes in an image region containing
                     some object and outputs a set of point locations that define
                     the pose of the object the [_dlib_pybind11.shape_predictor]

    Returns : rotated, the rotated image [numpy.ndarray]
              rotated_landmarks, an array containing the landmarks (68,2) [numpy.ndarray]
    """
    points, _ = faceLandmarks(image, predictor)


    
    left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
    right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
    points_left = [points[i] for i in left]
    points_right = [points[i] for i in right]

    eye_left = boundingBoxNaive(points_left)
    eye_right = boundingBoxNaive(points_right)

    # Get the eyes coordinates
    ex1 = eye_left[0][0]
    ey1 = eye_left[0][1]
    ew1 = eye_left[1][0] - ex1
    ed1 = eye_left[0][1] - ey1

    ex2 = eye_right[0][0]
    ey2 = eye_right[0][1]
    ew2 = eye_right[1][0] - ex2
    ed2 = eye_right[0][1] - ey2

    left_eye = (ex1,ey1,ew1,ed1)
    right_eye = (ex2,ey2,ew2,ed2)

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle =np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # Width and height of the image
    h, w = image.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)

    # Defining the rotation matrix M
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the cv2.warpAffine method
    rotated = cv2.warpAffine(image, M, (w, h))

    ones = np.ones((68,1), dtype=int)

    # Making the landmarks array 3D by adding one column in order to be able to perform
    # the matrix multiplication
    points3D = np.concatenate((points, ones), axis=1)
    rotated_landmarks = np.asarray([np.dot(M, landmark.T) for landmark in points3D])


    return rotated, rotated_landmarks


def cropResize(image, size, predictor):
    """
    Crops the dected face of the  

    Args: image, [numpy.ndarray]
          size, the targeted size of the resizing [int] 
          predictor, this object is a tool that takes in an image region containing
                     some object and outputs a set of point locations that define
                     the pose of the object the [_dlib_pybind11.shape_predictor]

    Returns : resized_cropped_image, the rotated and cropped image [numpy.ndarray]
              shape_cropped_resized, an array containing the landmarks (68,2) on the new 
                                     rotated and cropped image [numpy.ndarray]
    """

    shape, rects = faceLandmarks(image, predictor)

    # Get the bounding box of the face
 
    x, y, w, h  = rects[0], rects[1], rects[2], rects[3]


    # Open the bounding box in order to include all the landmarks of the face
    for ix in range(60,0,-1):
        if x-ix > 0:
            break
            
    for iy in range(60,0,-1):
        if y-iy > 0:
            break
            
    for iw in range(60,0,-1):
        if w+iw < image.shape[1]:
            break
            
    for ih in range(60,0,-1):
        if h-ih < image.shape[1]:
            break

    cropped_image = image[ y:h, x:w]
    resized_cropped_image = cv2.resize(cropped_image, (size,size), interpolation = cv2.INTER_AREA)

    shape_cropped_resized = []
    for landmark in range(len(shape)):
        x_old = shape[landmark][0]
        y_old = shape[landmark][1]
        (x_new, y_new) = getNewCoords(x_old, y_old, x, y, w, h, size, size )
        shape_cropped_resized.append((x_new, y_new))
    return resized_cropped_image, shape_cropped_resized



def hogsLandmarks(resizedCroppedImage, resizedCroppedLandmarks, folder):
    
    """
    Computes features of the image given the pretrained classifier using the hog values and the landmarks 

    Args: resizedCroppedImage, [numpy.ndarray]
          resizedCroppedLandmarks, an array containing the landmarks (68,2) [numpy.ndarray] 
          folder, the directory where the pretrained files are located [string]

    Returns : feature_cbd, features of the image (1, 1331) that can be used for classification of the AUs [numpy.ndarray]
    """
    _, pca_model, classifier, scaler = openFiles(folder)
    resizedCroppedLandmarks = np.array(resizedCroppedLandmarks)

    hull = ConvexHull(resizedCroppedLandmarks)
    mask = grid_points_in_poly(shape=np.array(resizedCroppedImage).shape, verts= list(zip(resizedCroppedLandmarks[hull.vertices][:,1], resizedCroppedLandmarks[hull.vertices][:,0])))
    resizedCroppedImage[~mask] = 0
    offline_hogs, hogs_im = hog(resizedCroppedImage, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)


    scaled_hogs = scaler.fit_transform(offline_hogs.reshape(-1,1))[0:5408].reshape(1,-1)
    pca_transformed_frame = pca_model.transform(scaled_hogs)
    feature_cbd = np.concatenate((pca_transformed_frame, resizedCroppedLandmarks.reshape(1,-1)), 1)
    return feature_cbd


def AUdetection(feature_cbd, folder):
    
    """
    Computes the AUs 

    Args: feature_cbd, the facial feature (1, 1331) used for the AU classification [numpy.ndarray]
          folder, the directory where the pretrained files are located [string]

    Returns : au_present, the AUs that are classified as present [list]
    """
    _, _, classifier, _ = openFiles(folder)
    combined_pred_aus = []
    for keys in classifier:
        au_pred = classifier[keys].predict(feature_cbd)
        combined_pred_aus.append(au_pred)

    au_array = [1,2,4,5,6,7,9,10,11,12,14,15,17,20,23,24,25,26,28,43]
    au_present = []
    for i in range(len(combined_pred_aus)):
        if combined_pred_aus[i] ==1:
            au_present.append(au_array[i])
    return au_present




def plotAUs(image, AUs, landmarks,rects):
    """
    Plots the AUs on the initial image 

    Args: image, [numpy.ndarray]
          AUs, the the AUs present in the face [list]
          landmarks, an array containing the landmarks (68,2) [numpy.ndarray] 
          rects, the coordinates (upperLeftX, upperLeftY, bottomleftX, bottomleftY)
                 of the face [list]

    Returns : nothing to return, the call of the function will diplay the plot 
    """

    distancex = image.shape[0] * (100/1024)
    distancey = image.shape[1] * (80/1024)
    plt.rcParams["figure.figsize"] = (45,8)
    fig, ax = plt.subplots()
    ax.imshow(image)

    # -------------------------------------------------
    if 1 in AUs:
        offsetbox = TextArea("Inner Brow Raiser", minimumdescent=False, textprops = dict(backgroundcolor = 'green'  ))
    else:
        offsetbox = TextArea("Inner Brow Raiser", minimumdescent=False, textprops = dict(backgroundcolor = 'red'  ))

    ab = AnnotationBbox(offsetbox, (landmarks[21][0],landmarks[21][1]),
                        xybox=( rects[0] - distancex, rects[1]- 2*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 2 in AUs:
        offsetbox = TextArea("Outer Brow Raiser ", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Outer Brow Raiser ", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, (landmarks[17][0],landmarks[17][1]),
                        xybox=( rects[0] - distancex, rects[1] - distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


    # -------------------------------------------------
    if 5 in AUs:
        offsetbox = TextArea("Upper Lid Raiser", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Upper Lid Raiser", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, (landmarks[37][0],landmarks[37][1]),
                        xybox=( rects[0] - distancex, rects[1]  ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 6 in AUs:
        offsetbox = TextArea("Cheek Raiser", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Cheek Raiser", textprops = dict(backgroundcolor = 'red'))
    ab = AnnotationBbox(offsetbox, ((landmarks[2][0]+ landmarks[31][0])/2, ( landmarks[2][1]+ landmarks[31][1])/2 ),
                        xybox=( rects[0] - distancex, rects[1] + distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 12 in AUs:
        offsetbox = TextArea("Lip Corner Depressor", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip Corner Depressor", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[48][0] ,  landmarks[48][1]),
                        xybox=( rects[0] - distancex, rects[1] + 2*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 20 in AUs:
        offsetbox = TextArea("Lip strecher", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip strecher", textprops = dict(backgroundcolor = 'red'))
    ab = AnnotationBbox(offsetbox, ( landmarks[59][0] ,  landmarks[59][1] ),
                        xybox=( rects[0] - distancex, rects[1]+ 3*distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 24 in AUs:
        offsetbox = TextArea("Lip pressor", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip pressor", textprops = dict(backgroundcolor = 'red'))
    ab = AnnotationBbox(offsetbox, ( landmarks[57][0] ,  landmarks[57][1] ),
                        xybox=(  rects[0] - distancex, rects[1]+ 4*distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 25 in AUs:
        offsetbox = TextArea("Lips part", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lips part", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[66][0] ,  (landmarks[66][1]+landmarks[62][1])/2 ),
                        xybox=( rects[0] - distancex, rects[1]+ 5*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    # -------------------------------------------------

    if 7 in AUs:
        offsetbox = TextArea("Lid Tightener", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lid Tightener", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[46][0] ,  landmarks[46][1] ),
                        xybox=( rects[2] + distancex, rects[1] - 2*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 43 in AUs:
        offsetbox = TextArea("Eyes closed", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Eyes closed", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[43][0] ,  landmarks[43][1] ),
                        xybox=( rects[2] + distancex, rects[1] - distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 9 in AUs:
        offsetbox = TextArea("Nose Wrinkler", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Nose Wrinkler", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[35][0] ,  landmarks[35][1] ),
                        xybox=( rects[2] + distancex, rects[1] ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 5 in AUs:
        offsetbox = TextArea("Upper Lip Raiser", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Upper Lip Raiser", textprops = dict(backgroundcolor = 'red'))
    ab = AnnotationBbox(offsetbox, ( landmarks[51][0] ,  landmarks[51][1] ),
                        xybox=( rects[2] + distancex, rects[1] + distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 28 in AUs:
        offsetbox = TextArea("Lip Suck", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip Suck", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[55][0] ,  landmarks[55][1] ),
                        xybox=( rects[2] + distancex , rects[1]+ 2*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 11 in AUs:
        offsetbox = TextArea("Nasolabial Deepener", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Nasolabial Deepener", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[54][0] ,  ((landmarks[54][1]+ landmarks[33][1])/2 )),
                        xybox=( rects[2] + distancex, rects[1]+3*distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 14 in AUs:
        offsetbox = TextArea("Dimpler", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Dimpler", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[54][0]+ 20 ,  landmarks[54][1]),
                        xybox=( rects[2] + distancex, rects[1]+ 4 * distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    if 12 in AUs:
        offsetbox = TextArea("Lip Corner Puller", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip Corner Puller", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[54][0]-10 ,  landmarks[54][1]),
                        xybox=( rects[2] + distancex, rects[1]+ 5 * distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

     # -------------------------------------------------
    if 23 in AUs:
        offsetbox = TextArea("Lip tighter", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Lip tighter", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[65][0] ,  landmarks[65][1] ),
                        xybox=( rects[2] -distancex, rects[1]+ 5*distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

    # -------------------------------------------------
    # -------------------------------------------------


    if 17 in AUs:
        offsetbox = TextArea("Chin Raiser", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Chin Raiser", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( (landmarks[57][0] + landmarks[8][0])/2 ,  (landmarks[57][1] + landmarks[8][1])/2),
                        xybox=( (rects[0]+ rects[3])/2 + distancex, rects[3]),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


    # -------------------------------------------------
    if 26 in AUs:
        offsetbox = TextArea("Jaw Drop", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Jaw Drop", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( landmarks[8][0] ,  landmarks[8][1]),
                        xybox=( (3*rects[0]+rects[2])/4, rects[3]+ distancey ),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


     # -------------------------------------------------
    if 4 in AUs:
        offsetbox = TextArea("Brow Lowerer", textprops = dict(backgroundcolor = 'green'))
    else:
        offsetbox = TextArea("Brow Lowerer", textprops = dict(backgroundcolor = 'red'))

    ab = AnnotationBbox(offsetbox, ( (landmarks[21][0] + landmarks[22][0])/2, landmarks[19][1]),
                        xybox=( (landmarks[21][0] + landmarks[22][0])/2 , landmarks[19][1] - 2*distancey),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)




# Helper functions

def openFiles(folder):
    """
    Function that helps to open the files of the pretained models.

    Args: folder, the directory where the pretrained files are located [string]

    Returns : predictor, this object is a tool that takes in an image region containing
                         some object and outputs a set of point locations that define
                         the pose of the object the [_dlib_pybind11.shape_predictor]
              pca_model, the pca pretrained model [sklearn.decomposition._pca.PCA]
              classifier, a dictionary including a sklearn.svm._classes.LinearSVC for each AU. [dict]
              scaler, the scaler [sklearn.preprocessing._data.StandardScaler]

    """

    predictor = dlib.shape_predictor(folder + "/shape_predictor_68_face_landmarks.dat")
    pca_model = joblib.load(folder + "/hog_pca_all_emotio.joblib")
    classifier = joblib.load(folder + "/svm_568.joblib")
    scaler = joblib.load(folder + "/hog_scalar_aus.joblib")
    return predictor, pca_model, classifier, scaler


def shape_to_np(shape, dtype="int"):
    """
    Transforms the output of the dlib in an array-type 
    
    Args: shape, the output of the dlib landmarks detection [_dlib_pybind11.full_object_detection]
    
    Returns: coords, the landmakrs in an array-type (68,2) [numpy.ndarray]
    """

    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def boundingBoxNaive(points):
    """
    Creates a bounding box that contains a list of coordinates
    
    Args: points, the landmakrs in an array-type (68,2) [numpy.ndarray]
    
    Returns: the coordinates of the bounding in the form :  [(upperLeftX, upperLeftY), (bottomRightX, bottYmRightX)], [list]
    """
    upperLeftX = min(point[0] for point in points)
    upperLeftY = min(point[1] for point in points)
    bottomRightX = max(point[0] for point in points)
    bottYmRightX = max(point[1] for point in points)

    return [(upperLeftX, upperLeftY), (bottomRightX, bottYmRightX)]


def getNewCoords(point_x,point_y,upperLeftX, upperLeftY, lowerrightX, lowerRightY, image_width,image_height):
    """
    Computes the new coordinates of a point in an image cropped based on a bounding box and rescaled.
    
    Args: point_x, the x-coordinate of the initial point [int]
          point_y, the y-coordinate of the initial point [int]
          upperLeftX, the x-coordinate of the upper left point of the bounding box [int]
          upperLeftY, the y-coordinate of the upper left point of the bounding box [int]
          lowerrightX, the x-coordinate of the lower tight point of the bounding box [int]
          lowerRightY, the y-coordinate of the lower tight point of the bounding box [int]
          image_width, the width of the resized image [int]
          image_height, the height of the resized image [int]
          
    Return:  (point_x,point_y), the coordinates of the new point [tuple] 
    """

    sizeX = lowerrightX - upperLeftX
    sizeY =  lowerRightY - upperLeftY
    centerX = (lowerrightX + upperLeftX)/2
    centerY = (lowerRightY + upperLeftY)/2

    offsetX = (centerX-sizeX/2)*image_width/sizeX
    offsetY = (centerY-sizeY/2)*image_height/sizeY

    point_x = point_x * image_width/sizeX - offsetX
    point_y = point_y * image_height/sizeY - offsetY
    return (point_x,point_y)

# not used in this version
def rotate(img, angle, landmarks):
    """
    Get the rotated image and the the set of landmarks given the angle of rotation
    
    Args: img, [numpy.ndarray]
          angle, the angle of rotation [float]
          landmarks, the landmakrs in an array-type (68,2) [numpy.ndarray]
    
    Returns: rotated_img, the rotated image [numpy.ndarray]
             rotated_landmarks, the rotated landmakrs in an array-type (68,2) [numpy.ndarray]
    """

    width, height = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv2.warpAffine(image, rotation, (width, height))
    ones = np.ones((68,1), dtype=int)
    points3D = np.concatenate((landmarks, ones), axis=1)
    rotated_landmarks = np.asarray([np.dot(rotation, landmark.T) for landmark in points3D])
    return rotated_img, rotated_landmarks


## References 
#  The eye detection method and the shape_to_np() are from https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6