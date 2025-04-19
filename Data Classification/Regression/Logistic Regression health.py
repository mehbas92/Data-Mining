import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

"""
dataset: https://datatab.net/statistics-calculator/regression?example=Medical_example_logistic_regression
Using Logistic regression to classify if specimen is diseased or not based on their smoking status:
The regression model takes the form of:
f(x) = 1/(1+e^(-x))
N.B: Use One Hot Encoding instead of label Encoder because the data is not ordinal and there is no relationship between the categories
"""
# Read the CSV file using pandas
def readFile(path):
    df = pd.read_csv(path)
    df.head() #file has headers
    return df

# Encode categorical attributes to numerical 
def encode_data(data):
    encoder = LabelEncoder()
    for header in data:
        data[header] = encoder.fit_transform(data[header])
    return data

# Split data into training and testing sets
def split_data(data, ratio):
    X = data.drop('Disease', axis=1)
    Y = data['Disease']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = ratio, random_state = None)

    return X_train, X_test, Y_train, Y_test

# Logistic Regression 
def LogisticReg(train_x, train_y, test_x):
    model = LogisticRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    return pred_y

# Scatter plot of Logistic regression
def histogram(data):
    crosstab = pd.crosstab(data['Smoker status'], data['Disease'], rownames=['Smoker status'], colnames=['Disease'])
    crosstab.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'])
    plt.title('Smoking vs Disease Status')
    plt.xlabel('Smoking (0=Non Smoker, 1=Smoker )')
    plt.ylabel('Count')
    plt.legend(title='Diseased (0=Diseased, 1=Not Diseased)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    path = 'C:\Data-Mining\datasets\datatab_health.csv'
    data = readFile(path)
    data = data.drop(['Age', 'Gender'], axis=1) #axis=1 represents column, axis=0 row
    data = encode_data(data)
    
    # Plot histogram to represent population based on smoking and disease status
    histogram(data)

    train_x, test_x, train_y, test_y = split_data(data, 0.20)
    pred_y = LogisticReg(train_x, train_y, test_x)


    print(confusion_matrix(test_y, pred_y))
    print("Accuracy:", accuracy_score(test_y, pred_y))

if __name__ == "__main__":
    main()