'''
The following code was implemented to train a decision tree model to identify food items using multiclass
classification technique. The model was then used to predict the type of food item shown in the pic and
the nutrition details. 
The trainfunc() function has the training code for the model. The model was trained to differenciate
amongst 5 different food items namely, apple, banana, bread, egg and grape. 
The food_nutrition() function has the nutrition values and depicts the values of the food item predicted
by the model.
'''

from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import confusion_matrix as cm, accuracy_score
import random
import pickle 

def trainfunc(image_dir):
    
    Categories = os.listdir(image_dir)
    Categories
    input_arr = []
    output_arr = []

    for i in Categories:
        print(f'Loading Category {i}')
        path = os.path.join(image_dir, i)
        for img in os.listdir(path):
            image_arr = imread(os.path.join(path, img))
            img_rsz = resize(image_arr, (150,150,3))
            input_arr.append(img_rsz.flatten())
            output_arr.append(Categories.index(i))
        print(f'Loaded Category {i} Succesfully')
        print(f"The code in output array is {Categories.index(i)}")
    
    flat_data = np.array(input_arr)
    target = np.array(output_arr)

    df = pd.DataFrame(flat_data)
    df['Target'] = target
    df.shape

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=47, stratify=y)

    print("Model is being trained")

    model = dtc(criterion='gini', splitter='best').fit(xtrain, ytrain)

    print(f'Model has been trained\n Testing')

    ypreds = model.predict(xtest)

    cm_report = cm(ytest, ypreds)
    acc = accuracy_score(ytest, ypreds)
    print(f"Accuracy of the model is {round((acc*100),2)}%")
    print(f"Confusion Matrix:\n{cm_report}")
    return model

image_dir = r"D:\AK\Career\Python\ProdigyInfotech\Task05_foodidentification\PRODIGY_ML_05\Food"
model = trainfunc(image_dir)
Categories = os.listdir(image_dir)



def food_nutrition(model, img, Categories):
    
    img = imread(os.path.join(pathu, rand_img))
    img_rsz = resize(img, (150,150,3)).flatten()


    val = Categories[model.predict(img_rsz.reshape(1,-1))[0]]
    print(f"The item in the picture is {val}")

    calorie_dict = {
        'Apple' : {'Calories': 52, 'Carbs': 14, 'Protein': 0.3, 'Fat': 0.2, 'Others': 'Vitamin C, Magnesium, Potassium'},
        'Banana' : {'Calories': 89, 'Carbs': 23, 'Protein': 1.1, 'Fat': 0.3, 'Others': 'Potassium, Vitamin C, Vitamin B6'},
        'Bread' : {'Calories': 265, 'Carbs': 49, 'Protein': 9, 'Fat':3.2, 'Others': 'Sodium, Calcium, Iron'},
        'Egg' : {'Calories': 155, 'Carbs': 1.1, 'Protein': 13, 'Fat': 11, 'Others': 'Vitamin D, Cobalamin, Iron'},
        'Grape': {'Calories': 67, 'Carbs': 17, 'Protein': 0.6, 'Fat': 0.4, 'Others': 'Vitamin C, Vitamin B6, Potassium'}
    }
    for i in calorie_dict[val]:
        print(f"{i}:{calorie_dict[val].get(i)}")
    print(f"(units for calories is kcal, rest are grams)\n the values are per 100 grams of the food item {val}")




pathu = r"D:\AK\Career\Python\ProdigyInfotech\Task05_foodidentification\PRODIGY_ML_05\Test"
rand_img = random.choice(os.listdir(pathu))
food_nutrition(model, rand_img, Categories)


model_pkl_file = r"D:\AK\Career\Python\ProdigyInfotech\Task05_foodidentification\PRODIGY_ML_05\model_dtc_pkl.pkl"

with open(model_pkl_file, 'wb') as file: 
    pickle.dump(model, file)


model_pkl_file = r"D:\AK\Career\Python\ProdigyInfotech\Task05_foodidentification\PRODIGY_ML_05\model_dtc_pkl.pkl"
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)