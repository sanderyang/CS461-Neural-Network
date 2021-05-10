import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#####Read in the file, and replace certain values with null.
erase_null_data = ['no_enrollment', 'Primary School', '<1', 'never']
data = pd.read_csv(r"C:\Users\Sandy\Documents\GitHub\Neural Network Py\aug_train.csv", na_values=erase_null_data)

####Drop these columns.
data.drop(columns=['city','enrollee_id'], inplace=True, axis=1)

####Replace null values in these columns with something pertinent.
data['education_level'].fillna("No Education", inplace=True)
data['major_discipline'].fillna("No Major", inplace=True)
data['enrolled_university'].fillna("No Enrollment", inplace=True)
data['gender'].fillna("Not Specified", inplace=True)
data['company_type'].fillna("Not Specified", inplace=True)
data['experience'].fillna(0, inplace=True)
data['last_new_job'].fillna(0, inplace=True)
data['company_size'].fillna("0", inplace=True)

####Replace certain weird values with ints -> make sure to convert to int, otherwise remains an object.
data.replace(to_replace=">20", value=21, inplace=True)
data["experience"] = data["experience"].astype(str).astype(int)
data.replace(to_replace=">4", value=5, inplace=True)
data["last_new_job"] = data["last_new_job"].astype(str).astype(int)
data.replace(to_replace="Oct-49", value="10-49", inplace=True)
data["company_size"] = data["company_size"].astype(str)

####Normalize the "experience", "last_new_job", "city_development_index" and "training_hours" columns to values 0.0 - 1.0.
data["experience"] = ((data["experience"] - data["experience"].min()) / (
        data["experience"].max() - data["experience"].min())) * 1
data["last_new_job"] = ((data["last_new_job"] - data["last_new_job"].min()) / (
        data["last_new_job"].max() - data["last_new_job"].min())) * 1
data["training_hours"] = ((data["training_hours"] - data["training_hours"].min()) / (
        data["training_hours"].max() - data["training_hours"].min())) * 1
data["city_development_index"] = ((data["city_development_index"] - data["city_development_index"].min()) / (
        data["city_development_index"].max() - data["city_development_index"].min())) * 1

#####One-encode the categorical columns in the dataset
one_hot_encoder = OneHotEncoder(sparse=False)

enrCol = pd.DataFrame(data['enrolled_university'])
eduCol = pd.DataFrame(data['education_level'])
majCol = pd.DataFrame(data['major_discipline'])
expCol = pd.DataFrame(data['relevent_experience'])
compCol = pd.DataFrame(data['company_type'])
genderCol = pd.DataFrame(data['gender'])
sizeCol = pd.DataFrame(data['company_size'])

one_hot_encoder.fit(expCol) # Hot one encode "has experience" column
exp_encoded = one_hot_encoder.transform(expCol)
exp_encoded = pd.DataFrame(data = exp_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(enrCol) # Hot one encode "enrolled" column
enrolled_encoded = one_hot_encoder.transform(enrCol)
enrolled_encoded = pd.DataFrame(data=enrolled_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(eduCol) # Hot one encode "education" column
educ_encoded = one_hot_encoder.transform(eduCol)
educ_encoded = pd.DataFrame(data=educ_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(majCol) # Hot one encode "major" column
maj_encoded = one_hot_encoder.transform(majCol)
maj_encoded = pd.DataFrame(data=maj_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(genderCol) # Hot one encode "gender" column
gender_encoded = one_hot_encoder.transform(genderCol)
gender_encoded = pd.DataFrame(data=gender_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(compCol) # Hot one encode "company_type" column
comp_encoded = one_hot_encoder.transform(compCol)
comp_encoded = pd.DataFrame(data=comp_encoded, columns=one_hot_encoder.categories_)
one_hot_encoder.fit(sizeCol) # Hot one encode "company_size" column
size_encoded = one_hot_encoder.transform(sizeCol)
size_encoded = pd.DataFrame(data=size_encoded, columns=one_hot_encoder.categories_)

####Reorganized the DataFrame: concatenating one hot coded categories & dropping unneeded columns.
data = pd.concat([data, enrolled_encoded, educ_encoded, maj_encoded, exp_encoded, gender_encoded, comp_encoded, size_encoded], axis=1)
data.drop(columns=['company_size', 'enrolled_university', 'education_level', 'major_discipline', 'relevent_experience', 'gender', 
                   'company_type'], inplace=True, axis=1)

####Exporting data into .csv file
data.to_csv('cleaned_data.csv', index=False)

print ('Finished cleaning up file....')

###########################################################################################
##### PASS IN CLEANED DATA INTO TENSORFLOW MODEL
###########################################################################################

#####Read in cleaned data file
#data = pd.read_csv("/content/cleaned_data.csv")

####Splitting up data into train, test, and validation)
targets = data.pop('target')
xTrain, xT, yTrain, yT = train_test_split(data.values, targets, test_size=0.3, shuffle=True)
xTest, xValidate, yTest, yValidate = train_test_split(xT, yT, test_size=0.5, shuffle=True)

opt = keras.optimizers.Adam(learning_rate=0.0005)
####Create the model function:
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


###################### TO RUN THE MODEL #############################
####Compile the model:
model = get_compiled_model()
model.fit(xTrain, yTrain, epochs=10, shuffle=True, batch_size=8, verbose=1)
results = model.evaluate(xTest, yTest)
print("Test loss, Test accuracy", results)


####################### TO SAVE THE MODEL ###########################
####Saving the model
#filepath = '/content/saved_model'
#save_model(model, filepath)


####################### TO LOAD THE MODEL ###########################
####Load the model - No need to retrain the model
#model = load_model(filepath, compile = True)


####################### PREDICTING ##################################
####Predicting values for the first 10 predictions
predictions = model.predict(xValidate[:10])

####Converting those predictions into target values of either 0 or 1
output = np.argmax(predictions[:10], axis = 1)

####Actual answers
validationDataFramAnswer = pd.DataFrame(yValidate)
validationDataFramAnswer = validationDataFramAnswer['target'].astype(str).astype(int)
validation_list = validationDataFramAnswer.tolist()

for x in range(0,10):
  print('Actual: ',validation_list[x], ' Guess: ', output[x])


####Converting the initial prediction values into probabilities
sm = tf.nn.softmax(predictions)
print('Column 1: No, Column 2: Yes',sm)
