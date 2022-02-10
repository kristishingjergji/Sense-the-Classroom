# For details: t https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/

import preprocess
from keras import backend
from sklearn.model_selection import train_test_split


def createTrainingTestingSet(participants, test_size ):
    finalImageALL = []
    finalLabelALL = []
    for person in participants:
        print(person)

        for session in preprocess.availableSessions(person,folder):
            print(session)
            AU, landmarks, image, imageLand = preprocess.landmarks_AU_CK(person, session, folder)

            ## Preprocessing 
            # 1. Align using landmarks of eyes
            image_aligned, landmarks_aligned = preprocess.alignFromLandmarks(image, landmarks)
            # 2. Crop image and resize 
            rect = preprocess.detectFace (image_aligned)
            croppedResizedImage = preprocess.cropResizeImage(image, rect, 96)
            
            # 3. Use grayscale (optional)
            #finalImage = cv2.cvtColor(croppedResizedImage, cv2.COLOR_BGR2GRAY)
            
            finalImage = croppedResizedImage

            finalImageALL.append(finalImage)
            finalImageALLresult = np.stack(tuple(finalImageALL), axis=0)

            labelEncoded = preprocess.labelsetCK(AU)
            finalLabelALL.append(labelEncoded)
            finallabelALLresult = np.stack(tuple(finalLabelALL), axis=0)

            X = finalImageALLresult
            y = finallabelALLresult

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42 )
        
    return X_train, X_test, y_train, y_test

 
# calculate fbeta score for multi-class/label classification
def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score