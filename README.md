# Steering-Track
## Behavioural Cloning
The goal of the project is to train a Deep Network to replicate the human steering behavior while driving, thus being able to drive autonomously on a simulator provided by Udacity. To this purpose, the network takes as input the frame of the frontal camera (say, a roof-mounted camera) and predicts the steering direction at each instant.

## Sample Demonstration:
https://youtu.be/v5y_K16DBD8

## The dataset
Data for this task can be gathered with the Udacity simulator itself. Indeed, when the simulator is set to training mode, the car is controlled by the human though the keyboard, and frames and steering directions are stored to disk. For those who want to avoid this process, Udacity made also available an "off-the-shelf" training set. For this project, I employed this latter.

## Visualizing training data
Here's how a typical sample looks like. We have three frames from different cameras as well as the associated steering direction.
![image](https://user-images.githubusercontent.com/105145597/168440224-9258c2e1-e583-40a1-b30f-1066ff7a0e81.png)

First things first, every frame is preprocessed by cropping the upper and lower part of the frame: in this way we discard information that is probably useless for the task of predicting the steering direction. Now our input frames look like these:

![image](https://user-images.githubusercontent.com/105145597/168440267-7619c501-d255-414e-873f-1d7c24f24cf1.png)

As we see, each frame is associated to a certain steering angle. Unfortunately, there's a huge skew in the ground truth data distribution: as we can see the steering angle distribution is strongly biased towards the zero.




Udacity training set is constituted by 8036 samples. For each sample, two main information are provided:
1. three frames from the frontal, left and right camera respectively
2. the corresponding steering direction
![image](https://user-images.githubusercontent.com/105145597/168439792-82e70a8f-615e-430e-ae56-b026a03f9ec5.png)

## Data Augmentation
Due to the aforementioned data imbalance, it's easy to imagine that every learning algorithm we would train on the raw data would just learn to predict the steering value 0. Furthermore, we can see that the "test" track is completely different from the "training" one form a visually point of view. Thus, a network naively trained on the first track would hardly generalize to the second one. Various forms of data augmentation can help us to deal with these problems.

Exploting left and right cameras
For each steering angle we have available three frames, captured from the frontal, left and right camera respectively. Frames from the side cameras can thus be employed to augment the training set, by appropriately correcting the ground truth steering angle. This way of augmenting the data is also reported in the NVIDIA paper.

Brightness changes
Before being fed to the network, each frame is converted to HSV and the Value channel is multiplied element-wise by a random value in a certain range. The wider the range, the more different will be on average the augmented frames from the original ones.

Normal noise on the steering value
Given a certain frame, we have its associated steering direction. However, being steering direction a continuous value, we could argue that the one in the ground truth is not necessarily the only working steering direction. Given the same input frame, a slightly different steering value would probably work anyway. For this reason, during the training a light normal distributed noise is added to the ground truth value. In this way we create a little more variety in the data without completely twisting the original value.

Shifting the bias
Finally, we introduced a parameter called bias belonging to range [0, 1] in the data generator to be able to mitigate the bias towards zero of the ground truth. Every time an example is loaded from the training set, a soft random theshold r = np.random.rand() is computed. Then the example i is discarded from the batch if steering(i) + bias < r. In this way, we can tweak the ground truth distribution of the data batches loaded. The effect of the bias parameter on the distribution of the ground truth in a batch of 1024 frames is shown below:

![image](https://user-images.githubusercontent.com/105145597/168439767-342cba1e-7315-480d-9425-18400e47a679.png)


## Network architecture
Network architecture is borrowed from the aforementioned NVIDIA paper in which they tackle the same problem of steering angle prediction, just in a slightly more unconstrained environment :-)

The architecture is relatively shallow and is shown below:

![nvidia_architecture](https://user-images.githubusercontent.com/105145597/168439719-691c5f1e-7373-4ff7-b077-70073d7e7742.png)

Input normalization is implemented through a Lambda layer, which constitutes the first layer of the model. In this way input is standardized such that lie in the range [-1, 1]: of course this works as long as the frame fed to the network is in range [0, 255].

The choice of ELU activation function (instead of more traditional ReLU) come from this model of CommaAI, which is born for the same task of steering regression. On the contrary, the NVIDIA paper does not explicitly state which activation function they use.

Convolutional layers are followed by 3 fully-connected layers: finally, a last single neuron tries to regress the correct steering value from the features it receives from the previous layers.

## Preventing overfitting
Despite the strong data augmentation mentioned above, there's still room for the major nightmare of the data scientis, a.k.a. overfitting. In order to prevent the network from falling in love with the training track, dropout layers are aggressively added after each convolutional layer (drop prob=0.2) and after each fully-connected layer (drop prob=0.5) but the last one.

## Training Details
Model was compiled using Adam optimizer with default parameters and mean squared error loss w.r.t. the ground truth steering angle. Training took a couple of hours on the GPU of my laptop (AMD Radeon Pro 5500M). During the training bias parameter was set to 0.8, frame brightness (V channel) was augmented in the range [0.2, 1.5] with respect to the original one. Normal distributed noise added to the steering angle had parameters mean=0, std=0.2. Frame flipping was random with probabililty 0.5.

## Testing the model
After the training, the network can successfully drive on both tracks. Quite surprisingly, it drives better and smoother on the test track with respect to the training track (at least from a qualitative point of view). I refer the reader to the demo video above for a visual evaluation of the model. This also comprises a visualization of the network's activations at different layer depth.

## Discussion and future works
In my opinion, these were the two main challenges in this project:

1. skew distribution of training data (strong bias towards 0)
2. relatively few training data, in one track only (risk of overfitting)
Both these challenges has been solved, or at least mitigated, using aggressive data augmentation and dropoout. The main drawback I notice is that now the network has some difficulties in going just straight: it tends to steer a little too much even when no steering at all is needed. Beside this aspect, the network is able to safely drive on both tracks, never leaving the drivable portion of the track surface.

There's still a lot of room for future improvements. Among these, would be interesting to predict the car throttle along with the steering angle. In this way the car would be able to mantain its speed constant even on the second track which is plenty of hills. Furthermore, at the current state enhancing the graphic quality of the simulator leads to worse results, as the network is not able to handle graphic details such as e.g. shadows. Collecting data from the simulator set with better graphic quality along with the appropriate data augmentation would likely mitigate this problem.





