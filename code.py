import pandas as pd
import numpy as np
import time


# Downloading the database to be used.  
train = pd.read_csv("./train.csv")
test  = pd.read_csv("./test.csv")

print("Features in Test  : ",test.shape[1])
print("Records  in Test  : ",test.shape[0])
print("Features in Train : ", train.shape[1])
print("Records  in Train : ",train.shape[0])

#Making a model of classificaiton that is independent of user that will be generalised as opposed to going for a user dependent data having higher accuracy
#but compromised on generalisation. In order to go with the former, I have removed the last two columns from the dataset. 
trainData  = train.drop(['subject','Activity'] , axis=1).values
trainLabel = train.Activity.values

testData  = test.drop(['subject','Activity'] , axis=1).values
testLabel = test.Activity.values

print("Shape of Train Data  : ",trainData.shape)
print("Shape of Train Label : ",trainLabel.shape)
print("Shape of Test Data : ",testData.shape)
print("Shape of Test Label : ",testLabel.shape)


#Converting the string data in label to categorical
from sklearn import preprocessing
from sklearn import utils

ltrain = preprocessing.LabelEncoder()
ltest = preprocessing.LabelEncoder()

trainLabel = ltrain.fit_transform(trainLabel)
testLabel  = ltest.fit_transform(testLabel)

print(np.unique(trainLabel))
print(np.unique(testLabel))
print("Shape of Train Label : ",trainLabel.shape)
print("Shape of Test Label  : ",testLabel.shape)
print(utils.multiclass.type_of_target(testLabel))



#Due to large number of features available, we have to go for a feature reduction in order to avoid overfitting.
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle

t0 = time.clock()
svc = SVC(kernel="linear")
# Calculating the accuracy scoring which is proportional to the number of correct classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
              scoring='accuracy')
# Shuffling the training data
np.random.seed(1)
print("Labels before Shuffle",testLabel[0:5])
testData,testLabel = shuffle(testData,testLabel)
trainData,trainLabel = shuffle(trainData,trainLabel)
print("Labels after Shuffle",testLabel[0:5])

rfecv.fit(trainData, trainLabel)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plotting features with cross validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# After an hour, the SVM model has been trained optimizing the features in the database. Using only these features
# will reduce the time of training of the model so used only 373 features instead of input of 561. 


print('Accuracy of the SVM model on test data is ', rfecv.score(testData,testLabel) )
# Getting the best features
best_features = []
for ix,val in enumerate(rfecv.support_):
    if val==True:
        best_features.append(testData[:,ix])


#The above yields an accuracy of approximately 97%. Following helps in visualization.
from pandas.tools.plotting import scatter_matrix
visualize = pd.DataFrame(np.asarray(best_features).T)
print(visualize.shape)
scatter_matrix(visualize.iloc[:,0:5], alpha=0.2, figsize=(6, 6), diagonal='kde')
