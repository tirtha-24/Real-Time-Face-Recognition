# **Real-Time-Face-Recognition**
This is a real time face recognition application able to recognize the person or face, streaming via web cam.
I have used pre trained model Keras-OpenFace which is an open source Keras implementation of the OpenFace (Originally Torch implemented) and is based on the CVPR 2015 paper FaceNet: A Unified Embedding for Face Recognition and Clustering by Florian Schroff, Dmitry Kalenichenko, and James Philbin at Google.
One shot learning paradigm is undertaken with an approach to implement the concept of siamese network. One shot learning allows us to save only one picture per person for recognizing.
Concept behind siamese network is that we calculate similarity between 2 images with their embeddings predicted through the pre-trained model, and recognize faces considering that similarity score.

## **Libraries Used**
1. Keras, for face recognition via FaceNet keras-implementation OpenFace.
2. OpenCV, for face detection via Haar-cascade classifier.

## **Theory**
### **One Shot Learning**
This is all about learning by seeing only one example. Algorithm is provided with only one or very few images per person. Model should be trained so that it can able to predict the correct 
face. This idea can be easily implemented with siamese network.

### **Siamese Network**
It is an architecture with two identical parallel neural networks, each taking a different input and whose outputs are combined to provide some prediction. Each image is passed through
the network to generate an embedding vector. Embedding vector is created for each single image stored in the database. When new face is detected for recognition, the new image is passed
through the same network to create an embedding and then this embedding is matched with all the embedding vectors stored for each image in the database and their similarity scores are calculated. Based on similarity 
score, the face is recognized.

### **Keras-Implementation of FaceNet, OpenFace**
OpenFace is a slight variation of the implementation of FaceNet's NN4 architecture, slightly smaller giving an improvement in execution time. Better to train when dataset is small. Input size corresponds to 96x96 pixels and output size corresponds to a 128-dimensional vector.

### **FaceNet**
Some important takeaways from the FaceNet Paper-
1. FaceNet directly learns a mapping frorm face images to a compact eucledian space where distances directly correspond to a measure of face similarity.
2. Triplet loss function is used for training the model.
3. Hard triplets are used for the sake of effectively learning the model. For the same, online generation of triplets are used where triplets are selected from mini-batches.
4. Model is trained with stochastic gradient descent with standard backpropagation and AdaGrad.
5. Two types of architectures are used, namely zeilur-fergus architecture and inception models.
6. All models proposed in this paper use ReLu activation function.
7. Validation rate and false accept rate are used for evaluation of the model.

### **Triplet Loss**
1. It involves an anchor image, a positive image and a negative image.
2. The loss function penalizes the model such that the distance between matching examples is reduced and the distance between the non-matching examples is increased.
3. A parameter called margin is used for the penalty.
4. Hard triplets are selected for ensuring fast convergence.

### **Viola Jones Algorithm-Haar Cascade Classifier**
OpenCV's cascade classifier based on Viola Jones algorithm is used to detect all the faces in the video-capture and then passing the detected face to siamese network for recognition.
The authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in the first five stages. In an image, most of the image is non-face region. For this they introduced the concept of Cascade of Classifiers. Instead of applying all 6000 features on a window, 
the features are grouped into different stages of classifiers and applied one-by-one. If a window fails the first stage, discard it. We don't consider the remaining features on it. If it passes, apply the second stage of features and continue the process. 
The window which passes all stages is a face region.
It takes 24x24 pixels window as input. Sliding window technique is used to select windows from the image and then the selected windows are rescaled until the size match the input size that is accepted by the cascade classifier and then that scaled image is fed into the classifier.

## **Components**
### **Data Creation**
A code section in my notebook captures 10 face images of the person. They all are stored in "images" folder with the name User_1 to User_10. Select a good captured image from the set of 10 images. Rename it with the name of person and delete rest of them. 
This image will be used for recognizing the the identity of the person using one shot learning.
### **Create Feature Embeddings**
This function generates 128 dimensional image embeddings of all the images stored in the "images" directory by feed forwarding the images to the pre-trained neural network, OpenFace.
### **Face Detection**
Haar-based cascade classifier is used to select the face regions in the video-capture.
### **Face Identification**
Face images are passed through the siamese network to generate the feature vector and then similarity score based on L2 eucledian distance is calculated for each stored vector and
the feature vector resembling to the least eucledian distance and satisfying a threshold is chosen and the name associated with that face is rendered along with a bounding box. Entire process of face detection
and recognition is processed within milliseconds or its better to say the application is real-time.

