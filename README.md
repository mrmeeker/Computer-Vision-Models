# Computer-Vision-Models
This is a storehouse for various CV models I'm making

Current Confusion Matrix picture is obtained by using the transfer learning model, trained for 20 epochs on the foo-11 dataset by
https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/

Current files:
models.py:
  Contains build_augment_layer function to create an augment layer using a TF sequential Model
  Contains an build_pretrained_model function to create a CV classifier using transfer learning from InceptionV3
  Contains a build_model function to create a basic Convolutional classifier
  Contains a train_model function, which compiles and trains the model
  Contains an evaluate_model function, which generates a sklearn based confusion matrix
  Contains a save_model and load_model functions
  
datahandler.py:
  Currently acts as a 'main' file, but will eventually be branched to just manipulate files, for example dividing the files into seperate folders
  Requires a cmd line argument for the location of the training data
  Optional cmd line argument for val, and test paths. Alternatively, can set the percentage (as float) of the training data to be used for validation (and evaluation if no evaluation path) default = 0.3 (30%)
  Makes the datasets
  Has a function to show 25 random images
  
To do:
  Add more functions to datahandler in order to set up the training enviroment (ie, seperate folders for the different classes based on a certain parameter in filename)
  Seperate the main functionality of datahandler to a real main file
  Perhaps use input() as opposed to cmd arguments. This could allow higher level functionality
  
  Add unet model to models.py
  Add GAN model to models.py
  Add style transfer GAN model to models.py
