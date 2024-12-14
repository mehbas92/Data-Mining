import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
Reference book: Data Mining: Theories, Algorithms and Examples by Nong Ye (Exercise 3.1)

Building a naïve Bayes classifier to classify a balloon as inflated or not based on its attributes such as
color, size, age, and act and then evaluating the classification performance of the model by computing 
what ratio of the records in the data set are correctly classified by the model.

Here is the intuition behind the model using Bayes Theorem:

P(C) = prior probability of class C (in this case either inflated or not)
P(xi|C) = likelihood of the occurrence of xi given C; Ex: likelihood of the balloon being yellow, given it is inflated
Product(P(xi|C)) = product of the likelihoods of all categorical attributes in the training dataset given the class is C

The posterior probability of each class is then calculated as follows:
P(C`) = P(C) * Product(P(xi|C))
To make a prediction for the test data select the class with the highest posterior probability 

All categorical values need to be assigned a numeric value for the scikit model to understand
'''

# Read the CSV file using pandas
def readFile(path):
    df = pd.read_csv(path)
    df.head() # file has headers
    return df

# Preprocessing: Convert categorical variables into numeric form using Label Encoding
def encode_data(data):
    encoder = LabelEncoder()
    for header in data:
        #data[header] = data[header].astype('category').cat.codes # Also Converts categorical columns to category codes (numeric values)
        data[header] = encoder.fit_transform(data[header])
    return data
    
# Split data into training and testing sets
def split_data(data, ratio):
    X = data.drop('Inflated', axis=1)
    Y = data['Inflated']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = ratio, random_state = None)

    return X_train, X_test, Y_train, Y_test

# Used when values are purely Categorical
def Categorical(X_train , Y_train, X_test):
    model = CategoricalNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    return Y_pred

# Gaussian naive bayes classifier is used when the values are continuous and the likelihoods follow a Gaussian distribution
def Gaussian(X_train , Y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    return Y_pred

def main():
    path = r'C:\Data Mining\datasets\Balloon.csv'
    data = readFile(path)
    data = encode_data(data)
    train_x, test_x, train_y, test_y = split_data(data, 0.2) # Using 20% of the dataset as test data

    pred_y = Categorical(train_x, train_y, test_x)
    y_pred = Gaussian(train_x, train_y, test_x)

    print("Accuracy using categorical NB:", accuracy_score(test_y, pred_y))
    print("Accuracy using Gaussian NB:", accuracy_score(test_y, y_pred))

    #print("Confusion Matrix:\n", confusion_matrix(test_y, pred_y))
    #print("Classification Report:\n", classification_report(test_y, pred_y))

if __name__ == "__main__":
    main()
