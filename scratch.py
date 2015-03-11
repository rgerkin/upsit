# Assumes that model5.py has been run

import numpy as np
import bbdp
from upsit import plt

def ROC(n_ctrl,means):
    n_subjects = means['r_subject'].shape[0]
    n_pd = n_subjects - n_ctrl
    for i in range(n_subjects):
        pass

def response_matrix(options=['responses'], exclude_subjects={}):
    ctrl = {}
    pd = {}
    subjects,tests = bbdp.load()
    exclude_subject_ids = []
    for key,value in exclude_subjects.items():
        exclude_subject_ids += [subject_id for subject_id,subject in subjects.items()\
                                if subject.__dict__[key]==value]
    subjects = {subject_id:subject for subject_id,subject in subjects.items()\
                if subject_id not in exclude_subject_ids}
    tests = [test for test in tests if test.subject.case_id not in exclude_subject_ids]
    scores = {}
    question_nums = range(1,41)
    possible_responses = [0,1,2,3]
    for test in tests:
        if test.subject.label == 'ctrl':
            ctrl[test.subject] = []
            x = ctrl[test.subject]
        if test.subject.label == 'pd':
            pd[test.subject] = []
            x = pd[test.subject]
        for q in question_nums:
            if 'responses' in options:
                response = []
                actual_response = test.response_set.responses[q].choice_num
                for possible_response in possible_responses:
                    if actual_response is possible_response:
                        x += [1]
                    else:
                        x += [0]    
            if 'responded' in options:
                x += [test.response_set.responses[q].choice_num is not None]
            if 'correct' in options:
                x += [int(test.response_set.responses[q].correct)]            
        if 'total_correct' in options:
            x += [sum([int(test.response_set.responses[q].correct) for q in question_nums])]
        if 'total_responded' in options:
            x += [sum([int(test.response_set.responses[q].choice_num is not None) for q in question_nums])]
        if 'gender' in options:
            x += [test.subject.gender]
        if 'expired_age' in options:
            x += [test.subject.expired_age]
     
    correct = np.array(list(ctrl.values()) + list(pd.values()),dtype='int')
    if 'num_each_type' in options:
        n = correct.shape[1]
        for i in range(4):
            correct = np.hstack((correct,correct[:,i:n:4].sum(axis=1).reshape(-1,1)))

    return correct,len(ctrl)

def classify(n_ctrl,data,alpha=1.0):
    """Naive bayes prediction of test results."""
    n_subjects = data.shape[0]
    n_pd = n_subjects - n_ctrl
    Y = np.array([0]*n_ctrl + [1]*n_pd)
    X = data
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    clf = MultinomialNB(alpha=alpha)
    
    scores = []
    for i in range(10000):
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
        clf.fit(X_train,Y_train)
        Y_pred = clf.predict(X_test)
        scores.append(metrics.accuracy_score(Y_test,Y_pred))
    scores = np.array(scores)
    print("\nClassification Accuracy: %.3f +/- %.3f" \
          % (scores.mean(),scores.std()/np.sqrt(i)))
    return clf,X,Y

def classify2(n_ctrl,data,clf=None):
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model, metrics
    from sklearn.cross_validation import train_test_split

    n_subjects = data.shape[0]
    n_pd = n_subjects - n_ctrl
    Y = np.array([0]*n_ctrl + [1]*n_pd)
    X = data
    
    # Models we will use
    if clf is None:
        logistic = linear_model.LogisticRegression()    
        rbm = BernoulliRBM(random_state=0, verbose=True)
        clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
        grid = GridSearchCV(logreg,{'C':[600]})
        grid.fit(X,Y)

        # Training Logistic regression
        #logistic_classifier = linear_model.LogisticRegression(C=100.0)
        #logistic_classifier.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        score = metrics.accuracy_score(Y_test,Y_pred)
        #score = classifier.score(X_test,Y_test)
        scores.append(score)

        '''
        print("Logistic regression using RBM features:\n%s\n" % (
            metrics.classification_report(
                Y_test,
                classifier.predict(X_test))))

        print("Logistic regression using raw pixel features:\n%s\n" % (
            metrics.classification_report(
                Y_test,
                logistic_classifier.predict(X_test))))
        '''

    print(np.mean(scores),np.std(scores),len(scores))
