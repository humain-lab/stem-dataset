# Stem-Dataset
Humain-Lab Stem datasets

Humain-Lab Stem datasets (Ba-Dataset and ImBa-Dataset)

The HUMAIN-Lab Stem datasets include to two different image datasets of grapevine stems. 

The datasets are designed to pose two different challenging problems: an imbalanced dataset [1], namely ImBa-Dataset, and a balanced dataset, namely Ba-Dataset. 

The ImBa-Dataset [1] comprises images of different quality and under varying illumination, displaying overlapping grape clusters or with dense leave coverage. This set is characterized as an imbalanced dataset since more pixels are belonging to leaves and grapes rather than to stems. 

The Ba-Dataset comprises of distinct stem images taken from the well-known Grape image database of Seng et al. [2]. 

Original images are augmented. From each original image, four new images are generated after subjected to four augmentation methods: flip (horizontal/vertical with probability 0.5), rotate (random degree between 90-280), shift (horizontal/vertical with probability 0.5) and shear (by -45/45 degrees with probability 0.5). The applied data augmentation procedure increased the initial training and testing data by a factor equal to 5. All images are scaled to 224×224 pixels size. 

Details regarding the two datasets are summarized in Table I. Indicative images of the two datasets are shown in Figure 1.

![stem-dataset-table](https://user-images.githubusercontent.com/26176656/112807357-7a7a9680-9080-11eb-964b-d6b909f9a557.png)

![stem-dataset-images](https://user-images.githubusercontent.com/26176656/112807397-86feef00-9080-11eb-8e95-4920987c9038.png)

The datasets refer to the work presented in [3]. In this work, a grape stem detection methodology in images is proposed. For this purpose, a regression convolutional neural network is applied to RGB images towards executing a stem segmentation task. Twelve Convolutional Neural Network (CNN) model architectures are investigated. Stem detection is tackled as a regression problem in a way to alleviate the imbalanced data phenomenon that may occur in vineyard images.

For both datasets are provided 6 folders of images (\*.png); three are for training (Train_\*) and three are for testing (Test_\*). The three folders include the RGB images (*_Images), Ground Truth images (*_Ground_Truth) and Bi-distance map images (*_Distance_Maps).

For anyone interested in reproducing the experiment presented in [3], in the Balanced_Dataset folder, may also find a script (Test_Models_For_Predictions.py) that reads the testing images and loads the UNET_mobilenetv2 (.h5) model for stem regression. 


Cite As:

Kalampokas, Τ.; Vrochidou, Ε.; Papakostas, G.A.; Pachidis, T.; Kaburlasos, V.G. Grape stem detection using regression convolutional neural networks. Comput. Electron. Agric. 2021.


References:
1. 	Kalampokas, T.; Tziridis, K.; Nikolaou, A.; Vrochidou, E.; Papakostas, G.A.; Pachidis, T.; Kaburlasos, V.G. Semantic Segmentation of Vineyard Images Using Convolutional Neural Networks. In 21st International Conference on Engineering Applications of Neural Networks (EANN 2020); 2020; pp. 292–303.
2. 	Seng, J.; Ang, K.; Schmidtke, L.; Rogiers, S. Grape image database Available online: https://researchoutput.csu.edu.au/en/datasets/grape-image-database (accessed on Jul 13, 2020).
3. 	Kalampokas, Τ.; Vrochidou, Ε.; Papakostas, G.A.; Pachidis, T.; Kaburlasos, V.G. Grape stem detection using regression convolutional neural networks. Comput. Electron. Agric. 2021.


