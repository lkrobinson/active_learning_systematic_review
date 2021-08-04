# active_learning_systematic_review
Source code for conducting active learning for systematic review screening


"""
Created on Fri May 22 13:53:58 2020

@author: louisarobinson
"""

import pandas as pd
import csv
from sklearn import preprocessing, linear_model, metrics
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter


with open('55_percent_texts.csv', newline = '') as df:
    studies = pd.read_csv(df, names=["id", "texts", "label"])
    studies["count"] = 1
    print(studies.groupby("label").count()["count"])
    
    # split into labelled and unlabelled dfs
    studies_lab = studies.dropna(subset = ["label"])
    studies_unlab = studies[~studies.label.isin(["$$titleabstractinclude", "$$titleabstractexclude"])]

    # Using labelled data
    lab_texts=[]
    labels=[]
    
    for study in studies_lab.index:
        lab_texts.append(str(studies_lab["texts"][study]))
        if "$$titleabstractexclude" in studies_lab["label"][study]:
            labels.append(0)
        else:
            labels.append(1)
    print('Studies for analysis: ', len(labels))
    print('Studies for analysis labelled include ',labels.count(1))
    print('Number of unlabelled studies: ', len(studies_unlab['id']))
    
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    labels = encoder.fit_transform(labels)
    
    # Create a word level tf-idf and use it to transform the training and
    # validation data
    tfidf_vect = \
        TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, stop_words='english')
    tfidf_vect.fit(lab_texts)
    texts_tfidf = tfidf_vect.transform(lab_texts) 

    # apply k-fold to split training and validation
    folds = KFold(n_splits=5, shuffle=True, random_state = 42)
    for train_index, test_index in folds.split(texts_tfidf, labels):
        x_train, x_test = texts_tfidf[train_index], texts_tfidf[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if(1): 
            # Resample to deal with class imbalance
            smote = SMOTE(random_state=42)
            x_train_res, y_train_res = smote.fit_resample(x_train, y_train) 
            x_train = x_train_res
            y_train = y_train_res
    
            # Recompute bias in training fold 
            print('Resampled dataset shape %s' % Counter(y_train_res))
        # Linear Classifier on Word Level TF IDF Vectors
        logreg = linear_model.LogisticRegression()
        logreg.fit(x_train, y_train)
        predictions = logreg.predict(x_test)
        accuracy = metrics.accuracy_score(predictions, y_test)
        print("LR, WordLevel TF-IDF\nAccuracy: ", accuracy)
        cf_matrix = metrics.confusion_matrix(predictions, y_test)
        print(cf_matrix)
        print(metrics.classification_report(y_test, predictions, target_names=['irrel', 'rel']))
    

    #create unlabelled texts list
    unlab_id = []
    unlab_texts = []
    for study in studies_unlab.index:
        unlab_texts.append(str(studies_unlab["texts"][study]))
        unlab_id.append(studies_unlab["id"][study])
    unlab_DF = pd.DataFrame()
    unlab_DF['text'] = unlab_texts
    unlab_DF['id'] = unlab_id
    
    id2abstract = dict(zip(unlab_id, unlab_texts))
    #Apply vectorizer to unlabelled data
    unlab_tfidf =  tfidf_vect.transform(unlab_texts)
    
    # apply classifier and establish probability of inclusion for unlabelled data
    pred_prob = logreg.predict_proba(unlab_tfidf)
    pred_incl = []
    for excl, incl in pred_prob:
        pred_incl.append(incl)
    print("Number of studies predicted labels: ", len(pred_incl))
    
    # Create dictionary with keys == study_ids, values == predicted_probability and export as csv

    dictionary = dict(zip(unlab_id, pred_incl))
    sorted_dict = sorted(dictionary.items(), key=lambda kv: kv[1], reverse= True)
    print("Number of rows in dictionary: ", len(sorted_dict))
    pred_outfile = csv.writer(open('pred_outfile.csv', 'w', newline='\n'))
    for ids, preds in sorted_dict:
        pred_outfile.writerow([ids, preds, id2abstract[ids]])
    


    
  
