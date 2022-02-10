import os
import cv2
import dlib
import numpy as np

class preprocess:
    def __init__(self, image):
        self.image = image

    def boundingBoxNaive (self):
        """ 
        Returns a list containing the bottom left and the top right 
        points in the sequence. It can be used to get the bounding box that a list of points are creating 
        """

        top_left_x = min(point[0] for point in self.points)
        top_left_y = min(point[1] for point in self.points)
        bot_right_x = max(point[0] for point in self.points)
        bot_right_y = max(point[1] for point in self.points)

        return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
    
    def alignFromLandmarks(self):
        """ 
        Returns aligned image and the angle of alignement given the landmarks 
        It requires boundingBoxNaive function """


        left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
        right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
        points_left = [self.points[i] for i in left]
        points_right = [self.points[i] for i in right]

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
        h, w = self.image.shape[:2]
        # Calculating a center point of the image
        # Integer division "//"" ensures that we receive whole numbers
        center = (w // 2, h // 2)
        # Defining a matrix M and calling cv2.getRotationMatrix2D method
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the cv2.warpAffine method
        rotated = cv2.warpAffine(self.image, M, (w, h))

        ones = np.ones((68,1), dtype=int)
        points3D = np.concatenate((points, ones), axis=1)
        rotated_landmarks = np.asarray([np.dot(M, landmark.T) for landmark in points3D])


        return rotated, rotated_landmarks
    
    

    


def getNewCoords(point_x,point_y,upperleftX, upperleftY, lowerrightX, lowerRightY, image_width, image_height):
    """ 
    Returns the new coordinates of a point (point_x,point_y), 
    in a cropped bounding box (upperleftX, upperleftY, lowerrightX, lowerRightY)
    and rescaled (image_width,image_height) image 
    """

    sizeX = lowerrightX - upperleftX
    sizeY =  lowerRightY - upperleftY
    centerX = (lowerrightX + upperleftX)/2
    centerY = (lowerRightY + upperleftY)/2

    offsetX = (centerX-sizeX/2)*image_width/sizeX
    offsetY = (centerY-sizeY/2)*image_height/sizeY

    point_x = point_x * image_width/sizeX - offsetX 
    point_y = point_y * image_height/sizeY - offsetY
    return (point_x,point_y)

        




def cropResize(image, rects, size, points):
    """ 
    Returns an image after cropping (given a rectagle, rects) and resizing in a given size  
    """

    for k, d in enumerate(rects):
        x = d.left()
        y = d.top()
        w = d.right()
        h = d.bottom()

    for ix in range(50,0,-1):
        if x-ix > 0:
            break
    for iy in range(50,0,-1):
        if y-iy > 0:
            break
    for iw in range(50,0,-1):
        if w+iw < image.shape[1]:
            break
    for ih in range(50,0,-1):
        if h-ih < image.shape[1]:
            break

    cropped_image = image[ y-iy:h+ih, x-ix:w+iw]
    cropped_resized_image= cv2.resize(cropped_image, (size,size), interpolation = cv2.INTER_AREA)
    pointsNew = points.copy()
    for i in range(len(points)):
        pointsNewx, pointsNewy = getNewCoords(points[i][0], points[i][1], x-ix, y-iy, w+iw, h+ih, size, size)
        pointsNew[i][0] = pointsNewx
        pointsNew[i][1] = pointsNewy

    return cropped_resized_image,pointsNew




def plotPoints(image,points):
    """ 
    Returns the plotted points in an image. The format of the points is a list of tuples of size 2  
    """
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    return image

def alignFromLandmarks(image, points):
    """ 
    Returns aligned image and the angle of alignement given the landmarks 
    It requires boundingBoxNaive function """


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
    # Defining a matrix M and calling cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the cv2.warpAffine method
    rotated = cv2.warpAffine(image, M, (w, h))
    
    ones = np.ones((68,1), dtype=int)
    points3D = np.concatenate((points, ones), axis=1)
    rotated_landmarks = np.asarray([np.dot(M, landmark.T) for landmark in points3D])


    return rotated, rotated_landmarks

def availableSessions(person,folder):
    """ Returns available sessions for a particular subject (= person) """
    
    imageFolder = folder + 'cohn-kanade-images/'
    personDirectory = imageFolder + person
    availableSessions = os.listdir(personDirectory)
    availableSessionsAll = [sess for sess in availableSessions if sess[0]!= '.']
    availableSessionsAll.sort()
    
    return availableSessionsAll
    


def landmarks_AU_CK(person, session, folder):
    """ Get the image """

    imageFolder = folder + 'cohn-kanade-images/'
    personDirectory = imageFolder + person
    personSession = imageFolder + person + '/' + session
    try:
        framesList = os.listdir(personSession)
        framesList.sort()
        frame = framesList[-1]
        imageDirectory = personSession + '/' + frame
        image =  cv2.imread(imageDirectory)
    except FileNotFoundError:
        availableSessions = availableSessions(person,folder)
        raise ValueError ('For this participant you can choose among the sessions', availableSessionsAll )

    """ Get the landmarks """

    landmarksFolder = folder + 'Landmarks/'
    landmarksDirectory = landmarksFolder + person + '/' + session
    landmarksList = os.listdir(landmarksDirectory)
    landmarksList.sort()
    landmarksFrame = landmarksList[-1]
    landmarksFileDirectory = landmarksDirectory + '/' + landmarksFrame
    with open(landmarksFileDirectory) as f:
        linesLandmakrs = f.readlines()


    """ Plot the landmarks on the image  """

    imageCopy = image.copy()
    landmarksArray = np.empty((68,2))
    for i in range(len(linesLandmakrs)):
        x1 = float(linesLandmakrs[i][3:16])
        y1 = float(linesLandmakrs[i][19:-1])
        landmarksArray[i][0] = x1
        landmarksArray[i][1] = y1
        cv2.circle(imageCopy, (int(x1), int(y1)), 3, (0, 0, 255), -1)

    """ Get the Action Units  """
    auFolder = folder + 'FACS/'
    ausDirectory = auFolder  + person + '/' + session + '/' 
    auList = os.listdir(ausDirectory)
    aus = auList[0]
    with open(ausDirectory + aus) as f:
        linesAus = f.readlines()
    ausPresent = []
    for i in range(len(linesAus)):
        ausPresent.append(int(float(linesAus[i][3:16])))

    return ausPresent,landmarksArray, image, imageCopy



def shape2np(shape, dtype="int"):
    """ 
    Returns the coordinates of the landmarks in 
    an array format given the output landmarks from dlib 
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def labelsetCK(AU):
    """ 
    Given a list of labels, it returns an encoded label array
    """
    AU = AU
    AUCK = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29, 30, 31,34,38,39,43, 44, 45, 54, 61,62, 63, 64]
    AUEncoded = np.zeros((39))
    for i in AU:
        index = AUCK.index(i)
        AUEncoded[index] = 1
    return AUEncoded


def boundingBoxNaive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    """
    top_left_x = min(point[0] for point in points)
    top_left_y = min(point[1] for point in points)
    bot_right_x = max(point[0] for point in points)
    bot_right_y = max(point[1] for point in points)

    return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]


def detectFace(image, predictor):
    """ 
    Returns aligned image and the angle of alignement given the landmarks 
    It requires boundingBoxNaive function """

    detector = dlib.get_frontal_face_detector() 
    detect = detector(image,1)
    shape= predictor(image,detect[0]) #the landmarks in another form
    points = shape_to_np(shape)
    rects = detector(image, 1)