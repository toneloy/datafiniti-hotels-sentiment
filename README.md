# Hotels Reviews Sentiment using Datafiniti's dataset

The goal of this project is to predict the rating of a hotel stay from a review in natural language using an Multilayer Perceptron (Neural Network). There are two components: a Jupyter Notebook to train the model and a Flask App to use/show the model.

## Jupyter Notebook
The notebook is pretty straight forward. 

- Load the data (which is included in the repo, so it's easier for anyone to use it),
- Separate data into training and test set
- Preprocess data (which I did using a bag-of-words approach)
- Design, train and save the network
- TODO: Use test set to assess model accuracy

## Flask App
This app can be runned in heroku, for example. It consist in a simple view at the homepage where you can write a review, and an _API_ endpoint on `/evaluate` where you can send a GET request.

## How to modify the model
You can possibly change the network architecture in that section of the notebook. Also, you might want to use the review body instead of the title, or use a bigger dictionary.
