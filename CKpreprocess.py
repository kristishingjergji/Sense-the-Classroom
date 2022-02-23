import os
import cv2
import numpy as np
import fee
from sklearn.model_selection import train_test_split

def landmarks_AU_CK(person, session, folder):
    """
    Opens and navigates through the files of the CK dataset

    Args: person, the code of the subject (person) e.g., S999 [str]
          session, the code of the session of the person e.g., 001 [str]
          folder, the directory where the CK dataset is located [str]

    Returns: au_present, the AUs that are present in the face given by the CK annotation [list]
            landmarks_array, the given landmakrs in an array-type (68,2) [numpy.ndarray]
            image, the image of the requested person in the requested session [numpy.ndarray]
            image_copy, the image of the requested person in the requested session with inluding the
                        plot of the landmarks[numpy.ndarray]
    """
    image_folder = folder + 'cohn-kanade-images/'
    person_directory = image_folder + person
    person_session = image_folder + person + '/' + session
    try:
        frames_list = os.listdir(person_session)
        frames_list.sort()
        frame = frames_list[-1]
        image_directory = person_session + '/' + frame
        image =  cv2.imread(image_directory)
    except FileNotFoundError:
        raise ValueError ('For this participant you can choose among the sessions', available_sessions(person,folder) )

    landmarks_folder = folder + 'Landmarks/'
    landmarks_directory = landmarks_folder + person + '/' + session
    landmarks_list = os.listdir(landmarks_directory)
    landmarks_list.sort()
    landmarks_frame = landmarks_list[-1]
    landmarks_file_directory = landmarks_directory + '/' + landmarks_frame
    with open(landmarks_file_directory) as f:
        lines_landmakrs = f.readlines()
    
    image_copy = image.copy()
    landmarks_array = np.empty((68,2))
    for i in range(len(lines_landmakrs)):
        x1 = float(lines_landmakrs[i][3:16])
        y1 = float(lines_landmakrs[i][19:-1])
        landmarks_array[i][0] = x1
        landmarks_array[i][1] = y1
        cv2.circle(image_copy, (int(x1), int(y1)), 3, (0, 0, 255), -1)

    ## Get the AUs
    au_folder = folder + 'FACS/'
    au_directory = au_folder  + person + '/' + session 
    au_list = os.listdir(au_directory)
    au = au_list[0]
    with open(au_directory + '/' + au) as f:
        lines_au = f.readlines()
    au_present = []
    for i in range(len(lines_au)):
        au_present.append(int(float(lines_au[i][3:16])))
    
    return au_present, landmarks_array, image, image_copy


def available_sessions(person,folder):
    
    """ Returns the available sessions for a particular subject
    
    Args: person, the code of the subject (person) e.g., S999 [str]
          folder, the directory where the CK dataset is located [str]
    
    Returns: availableSessionsAll, the available sessions [list]
    
    """
    
    image_folder = folder + 'cohn-kanade-images/'
    person_directory = image_folder + person
    available_sessions = os.listdir(person_directory)
    available_sessions_all = [sess for sess in available_sessions if sess[0]!= '.']
    available_sessions_all.sort()
    
    return available_sessions_all


def encode_au(au):
    """ 
    Encodes the AUs using a mapping on the possiblie AUs
    Args: au, the AUs present given by the CK annotation [list]
    
    Returns: au_encoded, the AUs present one hot encoded [list]
    
    """
    auCK = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,34,38,39,43,44,45,54,61,62,63,64]
    au_encoded = np.zeros((39))
    for i in au:
        index = auCK.index(i)
        au_encoded[index] = 1
    return au_encoded


def create_training_testing_set( folder, test_size, **kwargs ):
    """
    Creates the training and testing set using the CK dataset 
    
    Args: folder, the directory where the CK dataset is located [str]
          test_size, the percentage of the test set [float]
          ** dimension, the dimension of the image, 2 or 3 [int]
          
    Returns: X_train, X_test, y_train, y_test, arrays including the 
    
    
    """
    dimension = kwargs.get('dimension', None)
    participants = os.listdir(folder + 'cohn-kanade-images/')
    participants = [part for part in participants if part.lower().startswith('s')]
    participants.sort()
    image_final_all = []
    label_final_all = []
    for person in participants:
        #print(person)
        for session in available_sessions(person,folder):
            #print(session)
            au_present, landmarks_array, image, image_copy = landmarks_AU_CK(person, session, folder)

            ## Preprocessing 
            # 1. Align using landmarks of eyes
            image_aligned, landmarks_aligned = fee.align_from_landmarks(image, landmarks = landmarks_array)
            # 2. Crop image and resize 
            image_cropped_resized, landmarks_cropped_resized = fee.crop_resize(image_aligned, 96, landmarks = landmarks_aligned)
            if dimension == 3:
                image_final = image_cropped_resized
            elif dimension == 2:
                    image_gray = cv2.cvtColor(image_cropped_resized, cv2.COLOR_BGR2GRAY)
            else: 
                raise ValueError ('Invalid dimention. Choose between 3 and 2')
                
            
            

            image_final_all.append(image_final)
            image_final_all_result = np.stack(tuple(image_final_all), axis=0)

            label_encoded = encode_au(au_present)
            label_final_all.append(label_encoded)
            label_final_all_result = np.stack(tuple(label_final_all), axis=0)

            X = image_final_all_result
            y = label_final_all_result

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42 )
        
    return X_train, X_test, y_train, y_test