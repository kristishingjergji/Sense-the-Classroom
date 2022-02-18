# Sense the Classroom

This is the github repository of the project [Sense the Classroom](https://www.ou.nl/en/innovating-for-resilience-projects-sense-the-classroom) <br />
This project focuses on developing methods and means to ethically collect and use non-verbal cues of participants of online classrooms to assist teachers, students, and course coordinators by providing real-time and after-the-fact feedback of the students’ learning-centered affective states.


## 1. FEE
FEE is a tool that provides interpretation of the facial expressions utilising visualisation. <br />
### Libraries required:
* **`dlib`**
For the detection of the face and the landmarks.
* **`cv2`**
For image processing.
* **`matplotlib`**
For plotting.
* **`joblib`** 
For opening the pretained weights provided by `py-feat`
* `scipy` and `skimage`
For the ConvexHull method and the image processing respectively

### Files required
1. [Landmarks from `dlib`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Pretrained models from `py-feat`:  [pca](https://github.com/cosanlab/py-feat/releases/download/v0.1/hog_pca_all_emotio.joblib), [svm](https://github.com/cosanlab/py-feat/releases/download/v0.1/svm_568.joblib) and [hogs](https://github.com/cosanlab/py-feat/releases/download/v0.1/hog_scalar_aus.joblib)



