import csv
import itertools
from itertools import cycle

import pandas as pd
from matplotlib import pyplot as plt
import nltk
import numpy as np
import seaborn as sns

from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Conv2D,Flatten,MaxPooling2D
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
from sklearn import preprocessing, metrics
from scipy.sparse import coo_matrix, hstack

def ScatterPlot(data1,features_mean,df):
    #Color Labels - 0 is benign and 1 is malignant
    color_dic = {0:'red', 1:'yellow',2:'blue'}
    target_list = list(df['class'])
    colors = list(map(lambda x: color_dic.get(x), target_list))
    #Plotting the scatter matrix
    f, ax = plt.subplots(1, 1)  # plt.figure(figsize=(10,10))
    sm = pd.plotting.scatter_matrix(data1[features_mean], c= colors, alpha=0.4, figsize=((10,10)))
    plt.suptitle("Scatter matrix")
    plt.show()

def HeatMap(data,data_feature_names,df):
    # Arrange the data as a dataframe
    data1 = data
    data1.columns = data_feature_names
    # Plotting only 7 features out of 30
    # NUM_POINTS = 7
    features_mean = list(data1.columns[:])
    feature_names = data_feature_names[:]
    print(feature_names)
    f, ax = plt.subplots(1, 1)  # plt.figure(figsize=(10,10))
    sns.heatmap(data1[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
    # Set number of ticks for x-axis
    ax.set_xticks([float(n) + 0.5 for n in range(data_feature_names.__len__())])
    # Set ticks labels for x-axis
    ax.set_xticklabels(feature_names, rotation=25)
    # Set number of ticks for y-axis
    ax.set_yticks([float(n) + 0.5 for n in range(data_feature_names.__len__())])
    # Set ticks labels for y-axis
    ax.set_yticklabels(feature_names)
    plt.title("Correlation between various features")
    plt.show()
    plt.close()

    ScatterPlot(data1,features_mean,df)

def ROCPlot(y_test, y_score, n_classes):
    # ============================================================================
    # ROC Curve Setup
    # ============================================================================
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ============================================================================
    # Plot all ROC curves
    # ============================================================================
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MLP Over-sampling COVID-19 Data ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Zoomed in Over-sampling COVID-19 Data ROC')
    plt.legend(loc="lower right")
    plt.show()

def TrainMLP(X_train,y_train):
    # Best parameters found:
    #  {'activation': 'logistic', 'hidden_layer_sizes': (400, 400, 400), 'max_iter': 1000, 'solver': 'adam'}
    clf = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(10,), max_iter=1000,
                        solver='adam', learning_rate='adaptive')
    clf.fit(X_train, y_train)
    return clf

def ClassificationReport(clf,X_train,y_train,X_test, y_test,clf_predict):
    # Accuracy factors
    print('acc for training data: {:.3f}'.format(clf.score(X_train, y_train)))
    print('acc for test data: {:.3f}'.format(clf.score(X_test, y_test)))
    print('MLP Classification report:\n\n', classification_report(y_test, clf_predict))

def CrossValidation(X_train,y_train):
    models = []
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('Decision Tree', DecisionTreeClassifier()))
    models.append(('Perceptron', Perceptron(eta0=0.1, random_state=0, max_iter=100)))
    models.append(('MLP', MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(30,), max_iter=1000,
                                        solver='adam', learning_rate='adaptive')))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('10-fold cross-validation on Money dataset')
    plt.show()

def plot_confusion(cm, classes,normalize=False,title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.cm.Blues
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def FindConfusion(y_test,clf_predict,title):
    cm = confusion_matrix(y_test.argmax(axis=1), clf_predict.argmax(axis=1))
    plot_confusion(cm, classes=["Low", "Medium", "High"],
                   title=title)  # disp.figure_.suptitle("Confusion Matrix")
    # print("Confusion matrix:\n%s" % disp.confusion_matrix))
    plt.show()

def OperateGridSearch(X_train,y_train):
    #Grid Search
    parameter_space = {
        'hidden_layer_sizes': [(10,),(20,),(30,),(50,),(100,),(10,10)],
        'max_iter': [1000],
        'activation': ['logistic'],
        'solver': ['adam'],
        # 'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','invscaling','adaptive'],
    }
    clf = GridSearchCV(MLPClassifier(), parameter_space, n_jobs=-1)
    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    return clf

def CalculateMAE(X_train, X_val, y_train, y_val):
    my_mae = []
    layer_name = []
    layers = [10,15,20,25,30,35,40,50,100,150]

    # for 2 Layers
    for max_layer in layers:
        model = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(max_layer,), max_iter=1000, solver= 'adam',learning_rate = 'adaptive')
        model.fit(X_train, y_train)
        preds_val = model.predict(X_val)
        mae = mean_absolute_error(y_val.argmax(axis=1), preds_val.argmax(axis=1))
        my_mae.append(mae)
        layer_name.append(""+str(max_layer)+",")

    plt.plot(layer_name,my_mae)
    plt.title('Mean Absolute Error of Test set')
    plt.xticks(rotation=45)
    plt.show()

def main():
    RawTwitterDataset1 = pd.read_csv("TwitterDB_Money_06082020.csv")
    RawTwitterDataset2 = pd.read_csv("TwitterDB_vaccine_9-9-2020.csv")
    RawTwitterDataset3 = pd.read_csv("TwitterDB_COVID-19.csv")

    RawTwitterDataset = pd.concat([RawTwitterDataset1,RawTwitterDataset2])
    RawTwitterDataset = pd.concat([RawTwitterDataset,RawTwitterDataset3])

    print(RawTwitterDataset.head())

    print(RawTwitterDataset['full_text'].head())

    for text in RawTwitterDataset['full_text'].head():
        tokenized_text=word_tokenize(text)
        print(tokenized_text)

    RawTwitterDataset.info()

    print(RawTwitterDataset['class'].value_counts())

    FilterDataset = RawTwitterDataset[RawTwitterDataset['class'] != 'M']
    # FilterDataset = FilterDataset[FilterDataset['class'] != NULL]
    FilterDataset = FilterDataset[FilterDataset['class'].notnull()]
    # FilterDataset = FilterDataset[FilterDataset['class'] != np.nan]

    categorise = {False: 0, True: 1}
    FilterDataset["user_verified"] = FilterDataset["user_verified"].replace(categorise)
    categorise_class = {'0': 0, '1': 1, '2': 2, 2: 2, 1: 1, 0: 0}
    FilterDataset["class"] = FilterDataset["class"].replace(categorise_class)

    Sentiment_count=FilterDataset.groupby('class').count()
    plt.bar(Sentiment_count.index.values, Sentiment_count['Tweet_id'])
    plt.xlabel('Class')
    plt.ylabel('Number of Tweet')
    plt.xticks( np.arange(3),('fake contents', 'opinions', 'trustworthy contents'))
    plt.title('Money dataset')
    plt.show()

    # Class count
    count_class_1, count_class_2 ,count_class_0 = FilterDataset['class'].value_counts()

    print("count_class_0: ",count_class_0)
    print("count_class_1: ",count_class_1)
    print("count_class_2: ",count_class_2)

    # Divide by class
    df_class_0 = FilterDataset[FilterDataset['class'] == 0]
    df_class_1 = FilterDataset[FilterDataset['class'] == 1]
    df_class_2 = FilterDataset[FilterDataset['class'] == 2]



    df_class_0_over = df_class_0.sample(count_class_1, replace=True)
    df_class_2_over = df_class_2.sample(count_class_1, replace=True)
    df_test_over = pd.concat([df_class_0_over, df_class_1], axis=0)
    FilterDataset = pd.concat([df_test_over, df_class_2_over], axis=0)

    print('Random over-sampling:')
    print(FilterDataset['class'].value_counts())

    Sentiment_count=FilterDataset.groupby('class').count()
    plt.bar(Sentiment_count.index.values, Sentiment_count['Tweet_id'])
    plt.xlabel('Class')
    plt.ylabel('Number of Tweet')
    plt.xticks( np.arange(3),('fake', 'opinion', 'reliable'))
    plt.title('Random over-sampling Money dataset')
    plt.show()

    print(FilterDataset["user_verified"])

    SelectedFeature = FilterDataset[["user_followers_count","user_friends_count","user_verified","favorited","favorite_count","user_statuses_count","retweet_count","reply_count"]]
    Features = FilterDataset[["user_followers_count","user_friends_count","user_verified","favorited","favorite_count","user_statuses_count","retweet_count","reply_count","class"]]

    data = Features

    data_feature_names = [data.columns[i] for i in range(0, data.shape[1])]

    # HeatMap(data, data_feature_names, Features)

    enc = preprocessing.Normalizer()
    enc.fit(SelectedFeature)
    UserFeatures = enc.transform(SelectedFeature)

    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.tokenize import RegexpTokenizer
    #tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,ngram_range = (1,1),stop_words="english",tokenizer = token.tokenize)
    text_counts= cv.fit_transform(FilterDataset['full_text'])
    print(type(text_counts.toarray()))
    X = np.concatenate((text_counts.toarray(), UserFeatures),axis=1)

    FilterDataset.info()
    y = label_binarize(FilterDataset['class'], classes=[0, 1, 2])
    n_classes = y.shape[1]

    # In case that validation dataset is needed
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, random_state = 14, test_size = 0.20)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state = 14, test_size = 0.20)

    # Uncomment this to implement Mean Absolute Value
    CalculateMAE(X_train, X_val, y_train, y_val)

    X_train, X_test, y_train, y_test = train_test_split(X, FilterDataset['class'], test_size=0.20, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    clfMLP = TrainMLP(X_train, y_train)
    #

    # Uncomment to using Grid Search function
    # clf = OperateGridSearch(X_train, y_train)

    # Instead of targets, store output as prediction probabilities
    y_score = clfMLP.predict_proba(X_test)

    clf_predict = clfMLP.predict(X_test)

    clf_predict_on_train = clfMLP.predict(X_train)

    ClassificationReport(clfMLP, X_train, y_train, X_test, y_test, clf_predict)

    # CrossValidation(X_train,y_train)
    # Generate Confusion Matrix

    confusion_matrix
    FindConfusion(y_train, clf_predict_on_train, title="Combined Training Set Confusion matrix")

    ROCPlot(y_test, y_score, n_classes)


    X_train= np.reshape(X_train,(X_train.shape[0], 1, X_train.shape[1]))
    X_test= np.reshape(X_test,(X_test.shape[0], 1, X_test.shape[1]))

    print("shape: ", X.shape[1])

    model = Sequential()

    model.add(LSTM(1000, return_sequences=True, input_shape=(1,X.shape[1])))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss="mean_absolute_error", optimizer="adam", metrics= ['accuracy'])

    history = model.fit(X_train,y_train,epochs=40,batch_size=64,validation_data=(X_test,y_test),shuffle= True)

    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test,verbose=1)

    model.summary()

    print(score)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()