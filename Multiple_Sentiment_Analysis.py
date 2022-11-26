# Importing is done here for all required packages.
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from nltk.tokenize import word_tokenize


# This Converts the CSV tables into readable data arrays via numpy and pandas.
# Doing this allows the CSV data to be read and used by the various AI algorithms.
def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    df = df._get_numeric_data()
    numpy_array = df.to_numpy()
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels

# Using the load data function's return values, we can create six separate variables for the fields of both CSVs.
names, instances, labels = load_data_from_csv("reviews_Video_Games_training.csv")
testnames, testinstances, testlabels = load_data_from_csv("reviews_Video_Games_test.csv")

print("\n" "Algorithm results:" "\n")

# This is the Random Forest classifier, it utilises 300 estimators as more would be overfitting, 45 max features as the root of 2000 is 44.7
# and 300 jobs as it speeds up the prediction process as it allows for all 300 estimators to simultaneously function.
clf = RandomForestClassifier(n_estimators = 600,
                            max_features = 45,
                             n_jobs = 300)

# This is the random forest classifier being fitted with the instances and labels and then being used to predict labels from the test dataset.
clf.fit(instances, labels)
y_pred = clf.predict(testinstances)
# This is the classification report being printed for the random forest classifier, with 3 digits.
print("\n" "Random Forest:" "\n")
print(classification_report(testlabels, y_pred, digits=3))

# This is the support vector machine algorithm, it is fitted with both linear and rbf kernels in order to identify the kernel with the
# superior accuracy, as well as this, it can be used to determine what type of dataset the game reviews csv is.

# This is the linear kernel being created, fitted and turned into a prediction variable
linclass = svm.SVC(kernel='linear')
linclass.fit(instances, labels)
linprediction = linclass.predict(testinstances)

# This is the classification report being printed for the linear SVM classifier, with 3 digits.
print("\n" "Linear Support Vector Machine:" "\n")
print(classification_report(testlabels, linprediction, digits=3))

# This is the rbf kernel being created, fitted and turned into a prediction variable
nonlinclass = svm.SVC(kernel='rbf')
nonlinclass.fit(instances, labels)
nonlinprediction = nonlinclass.predict(testinstances)

# This is the classification report being printed for the nonlinear SVM classifier, with 3 digits.
print("\n" "Nonlinear Support Vector Machine:" "\n")
print(classification_report(testlabels, nonlinprediction, digits=3))

# This function receives the raw text tsv files and converts them into two lists: instances and labels.
# This allows for easy access to the individual sets of data for further processing.
def getdata(input_csv):
    instances = []
    labels = []
    for data in input_csv[0]:
        instances.append(data)
    for label in input_csv[1]:
        labels.append(label)
    return instances, labels

# this loads the raw training data for the algorithms as a dataframe.
df_train_data = pd.read_csv('reviews_Video_Games_training.raw.tsv', sep='\t', header=None)
# This loads the raw test data for the algorithms as a dataframe.
df_test_data = pd.read_csv('reviews_Video_Games_test.raw.tsv', sep='\t', header=None)
# Loads the sentiment lexicon as a pandas dataframe.
df_sentiment_lexicon = pd.read_csv('Games_senti_lexicon.tsv', delimiter='\t', header=None)
# Obtains a list of words from the lexicon dataframe.
sentiment_words = list(df_sentiment_lexicon[0])
# Obtains a list of scores from the lexicon dataframe.
sentiment_scores = list(df_sentiment_lexicon[1])

# Designates lists for further use.
predictions = []
highlabels = []
lowlabels = []
calcscore = []

# Calls the getdata function in order to create variables for the instances and labels of the test dataframe in list form.
instances, labels = getdata(df_test_data)
traininstances, trainlabels = getdata(df_train_data)
# This segment iterates through every single test label and sorts every review into a list of positive reviews and negative reviews via using the indexes of each label.
count = int(0)
index = int(0)
for label in trainlabels:
    if label == 1:
        highlabels.append(index)
    elif label == 0:
        lowlabels.append(index)
    index = index + 1

# This function receives a list of labels which are all positive or all negative, it then tokenizes the review associated with that label and returns a sentiment score for that review using the lexicon.
# Once this has been performed for all reviews of that type, the average score for that review type is obtained via dividing the sum of every review sentiment score by the number of reviews.
def labelcalc(labels):
    lc = int(0)
    sentiment = float(0)
    for label in labels:
        lc = lc + 1
        text = instances[label]
        token = word_tokenize(text)
        for word in token:
            if word in sentiment_words:
                index = sentiment_words.index(word)
                index = index - 1
                score = float(sentiment_scores[index])
                sentiment = sentiment + score
    calc = sentiment / lc
    return calc

# This calls the labelcalc function to obtain the average score of all the positive reviews and all the negative reviews.
goodaverage = labelcalc(highlabels)
badaverage = labelcalc(lowlabels)

# This code is the main prediction segment. It obtains every review and for each review, tokenizes the words, cross-referencing them with the lexicon via indexes to obtain a cumulative sentiment for it.
# Once a review has been assigned a sentiment score, it is then compared to the average good score and average bad score in order to predict its label.
# Finally, the list of predicted labels is appended with the processed review.
for feature in instances:
    sentiment = int(0)
    token = word_tokenize(feature)
    for word in token:
        if word in sentiment_words:
            index = sentiment_words.index(word)
            index = index - 1
            score = float(sentiment_scores[index])
            sentiment = sentiment + score
    if sentiment >= goodaverage:
        predictions.append(1)
    elif sentiment <= badaverage:
        predictions.append(0)
    elif sentiment > badaverage * 1.25:
        predictions.append(1)
    else:
        predictions.append(0)

# This classification report displays the f-measure score among other metrics relating to the accuracy of this lexicon based algorithm.
print(classification_report(labels, predictions, digits=3))
