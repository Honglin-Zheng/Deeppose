# 3D Human Pose Estimation via Deep Neural Network

## Methodology
The prediction model adapts the [VGG16](https://arxiv.org/abs/1409.1556) architecture. Based on the [pre-trained model](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) that was used by the VGG team in the ILSVRC-2014 competition, we are using the pre-trained weight for all the convolutional layers to extract deep features from the images and fine tuning the last two dense layers on the [FLIC dataset](http://bensapp.github.io/flic-dataset.html). 

## Dependency
[Keras:Deep Learning library for Theano and TensorFlow](https://keras.io/)

## Sample Output
Some sample outputs on the FLIC dataset. Skeleton in green is groundtruth, while the red one is prediction from the model.

![image](https://cloud.githubusercontent.com/assets/11875272/20765529/ba69fbcc-b700-11e6-8b8f-24e0b6138ef0.png)![image](https://cloud.githubusercontent.com/assets/11875272/20765528/ba69eeca-b700-11e6-8a84-25a0ae32c3ad.png)![image](https://cloud.githubusercontent.com/assets/11875272/20765531/ba6ccece-b700-11e6-8d11-b9d371cdeb59.png)
![image](https://cloud.githubusercontent.com/assets/11875272/20765530/ba6b4a40-b700-11e6-919f-44949a85d89d.jpg)![image](https://cloud.githubusercontent.com/assets/11875272/20765532/ba713400-b700-11e6-8f05-866eff7ae022.jpg)![image](https://cloud.githubusercontent.com/assets/11875272/20765533/ba748574-b700-11e6-957e-bc528960cd40.jpg)

## Future Work
We are actively working on accomodating the model to perform 3D pose estimation.
