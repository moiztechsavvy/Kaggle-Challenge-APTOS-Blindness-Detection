# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

# %% [markdown]
# $x^2$

# %%
traindata2015 = pd.read_csv('data/trainLabels15.csv')
testdata2015 = pd.read_csv('data/testLabels15.csv')
traindata2019 = pd.read_csv('data/trainLabels19.csv')
testdata2019 = pd.read_csv('data/testLabels15.csv')
mesidor_data = pd.read_csv('data/messidor_data.csv')


# %%
mesidor_data.head()


# %%
path_to_messidor = 'data/messidor-2'
path_to_test_19 = 'data/resized test 19'
path_to_test_15 = 'data/resized test 15'
path_to_train_19 = 'data/resized train 19'
path_to_train_15 = 'data/resized train 15'


# %%
traindata2015.columns


# %%
Training_DataFrame  = pd.DataFrame({
    'image_id': [],
    'diagnosis': [],
    'dataSet': []
})


# %%
mesidor_data = mesidor_data.drop(columns=['adjudicated_dme', 'adjudicated_gradable'])
traindata2015 = traindata2015.rename(columns ={"image": "image_id", "level": "diagnosis"})
testdata2015 = testdata2015.rename(columns ={"image": "image_id", "level": "diagnosis"})
testdata2019 = testdata2019.rename(columns ={"id_code": "image_id"})
traindata2019 = traindata2019.rename(columns ={"id_code": "image_id"})
mesidor_data = mesidor_data.rename(columns ={"image_id": "image_id", "adjudicated_dr_grade": "diagnosis"})







# %%
traindata2019['dataSet'] = path_to_train_19
mesidor_data['dataSet'] = path_to_messidor
traindata2015['dataSet'] = path_to_train_15
testdata2015['dataSet'] = path_to_test_15
testdata2019['dataSet'] = path_to_test_19


# %%
traindata2019.tail()


# %%
Training_DataFrame = Training_DataFrame.append(traindata2019, sort=False,ignore_index=True)
Training_DataFrame = Training_DataFrame.append(traindata2015, sort=False,ignore_index=True)
Training_DataFrame = Training_DataFrame.append(mesidor_data, sort=False,ignore_index=True)
Training_DataFrame = Training_DataFrame.append(testdata2015, sort=False,ignore_index=True)


# %%
Index_of_null_vals = Training_DataFrame[Training_DataFrame['diagnosis'].isnull()].index


# %%
Training_DataFrame.drop(Index_of_null_vals)


# %%
import matplotlib.pyplot as plt;
plt.bar([0,1,2,3,4],  list(Training_DataFrame['diagnosis'].value_counts()),align='center', alpha=0.5)


# %%
d0,d1,d2,d3,d4 = Training_DataFrame['diagnosis'].value_counts()


# %%
d2


# %%
_d0 = Training_DataFrame[Training_DataFrame['diagnosis'] == 0].sample(d2)
_d1 = Training_DataFrame[Training_DataFrame['diagnosis'] == 1].sample(d2,replace=True)
_d2 = Training_DataFrame[Training_DataFrame['diagnosis'] == 2].sample(d2,replace=True)
_d3 = Training_DataFrame[Training_DataFrame['diagnosis'] == 3].sample(d2,replace=True)
_d4 = Training_DataFrame[Training_DataFrame['diagnosis'] == 4].sample(d2,replace=True)


# %%
Balance_Df = _d0
Balance_Df = Balance_Df.append(_d1)
Balance_Df = Balance_Df.append(_d2)
Balance_Df = Balance_Df.append(_d3)
Balance_Df = Balance_Df.append(_d4)


# %%
list(Balance_Df['diagnosis'].value_counts())


# %%
from sklearn.utils import shuffle

Balance_Df = shuffle(Balance_Df)


# %%
import matplotlib.pyplot as plt;
plt.bar([0,1,2,3,4], list(Balance_Df['diagnosis'].value_counts()),align='center')


# %%
Training_DataFrame['diagnosis'].value_counts()


# %%
Balance_Df['diagnosis'].value_counts()


# %%
Balance_Df.head()


# %%
Training_DataFrame['image_id']


# %%



# %%
Balance_Df["Path_to_image"] = "data/preprocessed_train"+ "/" + Balance_Df['image_id'] + ".png"


# %%
Balance_Df.head()
Balance_Df.drop('Usage',axis=1)
Balance_Df.drop('dataSet',axis=1)


# %%
Balance_Df.to_csv("smallerdataset.csv")

