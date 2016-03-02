import nbformat
from IPython.display import Image,display,HTML

import numpy as np
import pandas
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.cross_validation import cross_val_score,LeaveOneOut,\
                                     ShuffleSplit,LabelShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,\
                               OutputCodeClassifier
from sklearn.preprocessing import MultiLabelBinarizer,Imputer
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import seaborn as sns
import bs4

import bbdp
from upsit import plt

def get_response_matrix(kind, options=['responses'], exclude_subjects={}):
    responses = {}
    subjects,tests = bbdp.load(kind)
    exclude_subject_ids = []
    for key,value in exclude_subjects.items():
        exclude_subject_ids += [subject_id for subject_id,subject in subjects.items()\
                                if subject.__dict__[key]==value]
    subjects = {subject_id:subject for subject_id,subject in subjects.items()\
                if subject_id not in exclude_subject_ids}
    tests = [test for test in tests if test.subject.case_id not in exclude_subject_ids]
    question_nums = range(1,41)
    possible_responses = [0,1,2,3]
    for test in tests:
        x = []
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
        if 'fraction_correct' in options:
            x += [sum([int(test.response_set.responses[q].correct) for q in question_nums])/40.0]
        if 'total_responded' in options:
            x += [sum([int(test.response_set.responses[q].choice_num is not None) for q in question_nums])]
        if 'gender' in options:
            x += [test.subject.gender]
        if 'dementia' in options:
            x += [test.subject.dementia]
        if 'expired_age' in options:
            x += [test.subject.expired_age]
        responses[test.subject] = x

    ctrl = []
    for key in sorted(responses.keys(),key=lambda x:x.case_id):
        ctrl += [subjects[key.case_id].label == 'ctrl']
    
    responses = np.array(list(responses[key] for key in \
                       sorted(responses.keys(),key=lambda x:x.case_id)),dtype='float')
    
    
    if 'num_each_type' in options:
        n = responses.shape[1]
        for i in range(4):
            responses = np.hstack((responses,responses[:,i:n:4].sum(axis=1).reshape(-1,1)))

    return responses,ctrl


def get_labels(kind, exclude_subjects={}):
    labels = {}
    subjects,tests = bbdp.load(kind)
    exclude_subject_ids = []
    for key,value in exclude_subjects.items():
        exclude_subject_ids += [subject_id for subject_id,subject in subjects.items()\
                                if subject.__dict__[key]==value]
    subjects = {subject_id:subject for subject_id,subject in subjects.items()\
                if subject_id not in exclude_subject_ids}
    tests = [test for test in tests if test.subject.case_id not in exclude_subject_ids]
    n_ctrl = 0
    for test in tests:
        x = []
        if test.subject.label == 'ctrl':
            n_ctrl += 1
        if kind == 'dugger':
            x += [test.subject.label == 'pd']
        #x += [test.subject.expired_age-90]
        #x += [test.subject.gender]
        #if hasattr(test.subject,'dementia'):
        #    x += [test.subject.dementia]
        #x += [test.subject.stint]
        if hasattr(test.subject,'other'):
            x += test.subject.other
        x = [float(_) for _ in x]
        labels[test.subject] = x
    
    labels = [labels[key] for key in \
                       sorted(labels.keys(),key=lambda x:x.case_id)]
    return np.array(labels)


def summarize(X_responses,ctrl):
    print("The loaded matrix has shape (%d x %d), and there are %d controls" % (X_responses.shape[0],
                                                                            X_responses.shape[1],
                                                                            np.sum(ctrl)))


def plot_total_correct_cumul(X_total_correct,ctrl):
    X_ctrl = X_total_correct[ctrl == True,0]
    X_pd = X_total_correct[ctrl == False,0]
    plt.plot(sorted(X_ctrl),np.linspace(0,1,len(X_ctrl)),'k',label='Control')
    plt.plot(sorted(X_pd),np.linspace(0,1,len(X_pd)),'r',label='PD')
    plt.xlabel('Total Correct')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc=2)


def cross_validate(clf,X,Y,cv,kind):
    mean = cross_val_score(clf,X,Y,cv=cv).mean()
    print("Cross-validation accuracy for %s is %.3f" % (kind,mean))


def get_p_parks(mnb,loo,X,Y):
    p_parks = []
    for i, (train, test) in enumerate(loo):
        mnb.fit(X[train], Y[train])
        p_parks.append(mnb.predict_proba(X[test])[:,1][0])
    return p_parks


def get_roc_curve(Y,p):
    fpr, tpr, thresholds = roc_curve(Y, p)
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc


def plot_roc_curve(Y,p_parks_tc,p_parks_r):
    fpr_tc,tpr_tc,roc_auc_tc = get_roc_curve(Y,p_parks_tc)
    fpr_r_mnb,tpr_r_mnb,roc_auc_r_mnb = get_roc_curve(Y,p_parks_r)
    plt.plot(fpr_tc, tpr_tc, lw=2, color='gray', label='AUC using Total Correct = %0.2f' % (roc_auc_tc))
    #plot(fpr_r_bnb, tpr_r_bnb, lw=2, color='r', label='Responses area = %0.2f' % (roc_auc_r_bnb))
    plt.plot(fpr_r_mnb, tpr_r_mnb, lw=2, color='g', label='AUC using individual responses = %0.2f' % (roc_auc_r_mnb))
    plt.xlabel('False Positive Rate')#, fontsize='large', fontweight='bold')
    plt.ylabel('True Positive Rate')#, fontsize='large', fontweight='bold')
    plt.title('ROC curves')#, fontsize='large', fontweight='bold')
    plt.xticks()#fontsize='large', fontweight='bold')
    plt.yticks()#fontsize='large', fontweight='bold')
    plt.legend(loc="lower right")


def plot_roc_curves(Y,p,ax=None,label='full',title='ROC Curves'):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    colors = {'basic':'gray','total':'pink','all':'red'}
    for key in p:
        fpr,tpr,auc = get_roc_curve(Y[key],p[key])
        color = 'red' if key not in colors else colors[key]
        ax.plot(fpr, tpr, lw=2, color=color, 
                label={'full':'AUC using\n%s = %0.2f' % (key,auc),
                       'sparse':key}[label])
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    ax.set_xlabel('False Positive Rate')#, fontsize='large', fontweight='bold')
    ax.set_ylabel('True Positive Rate')#, fontsize='large', fontweight='bold')
    ax.set_title(title)#, fontsize='large', fontweight='bold')
    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        pass
        #label.set_fontsize('large')
        #label.set_fontweight('bold')
    ax.legend(loc="lower right")


def roc_data(X,Y,clf,n_iter=50,test_size=0.1):
    if n_iter is None and test_size is None:
        cv = LeaveOneOut(Y.shape[0])
    else:
        cv = ShuffleSplit(Y.shape[0],n_iter=n_iter,test_size=test_size)
    n_labels = Y.shape[1]
    Y_cv = {i:[] for i in range(n_labels)}
    p = {i:[] for i in range(n_labels)}
    p_1 = {i:[] for i in range(n_labels)}
    p_0 = {i:[] for i in range(n_labels)}
    for train, test in cv:
        clf.fit(X[train,:], Y[train,:])
        Y_predicted = clf.predict_proba(X[test,:])
        for i in range(Y.shape[1]):
            if type(Y_predicted) is list:
                p_ = 1 - Y_predicted[i][:,0]
            else:
                p_ = Y_predicted[:,i]
            Y_cv[i] += list(Y[test,i])
            p[i] += list(p_)
            p_1[i] += list(p_[np.where(Y[test,i]==1)[0]])
            p_0[i] += list(p_[np.where(Y[test,i]==0)[0]])
    return Y_cv, p, p_1, p_0


def violin_roc(data):
    plt.figure(figsize=(15, 15))
    sns.set_context("notebook", font_scale=2.5, 
                    rc={"lines.linewidth": 1.5, 'legend.fontsize': 20})
    sns.violinplot(x='Predicted Probability', y='Diagnosis', hue='Outcome', 
                   data=data, split=True, inner="quart", 
                   palette={'--': "y", '+': "b"}, orient='h', width=1.0, 
                   scale='area',#count
                   order=['VaD','Tauopathy NOS','AG','DLB','LB','ILBD',
                          'PD','AD','Parkinsonism NOS','PSP'])
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    plt.setp(ltext, fontsize=24)    # the legend text fontsize
    plt.xlim(0,1)
    sns.despine(left=True)


def plot_roc_curves_with_ps(Y,Y_cv,Xs,p0,p1,p,pathological,diagnoses=None):
    if diagnoses is None:
        diagnoses = pathological.columns.values
    fig,ax = plt.subplots(len(diagnoses),4)
    sns.set_context("notebook", font_scale=1, 
                    rc={"lines.linewidth": 1.5, 
                    'legend.fontsize': 12})
    fig.set_size_inches(15,2*len(diagnoses))
    for i in range(len(diagnoses)):
        diagnosis = diagnoses[i]
        ix = list(pathological.columns.values).index(diagnosis)
        if diagnoses is not None and diagnosis not in diagnoses:
            continue
        for j,key in enumerate(['basic','total','all']):
            X = Xs[key]
            if len(p1[key][ix]) and len(p0[key][ix]):
                ax[i,j].hist(p1[key][ix],bins=1000,range=(0,1),color='r',
                             normed=True,cumulative=True,histtype='step')
                ax[i,j].hist(p0[key][ix],bins=1000,range=(0,1),color='k',
                             normed=True,cumulative=True,histtype='step')
                ax[i,j].set_title(diagnosis)
            if i==ax.shape[0]-1:
                ax[i,j].set_xlabel('Predicted p(pathology)')
            if j==0:
                ax[i,j].set_ylabel('Cumulative fraction')
            ax[i,j].set_xlim(0,1)
            ax[i,j].set_ylim(0,1)
        plot_roc_curves({key:Y_cv[key][ix] for key in Y_cv},
                        {key:p[key][ix] for key in p},
                        ax=ax[i,3],label='sparse')

    fig.tight_layout()


def plot_just_rocs(Y_clean,ps,ys,imps):
    fig,axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(10,10))
    y = {}
    p = {}
    for i in range(len(list(Y_clean))):
        ax = axes.flat[i]
        for imp in imps:
            pimp = ps[imp][i,:,:].ravel() # Unravel all predictions of test data over all cv splits.  
            yimp = ys[imp][i,:,:].ravel() # Unravel all test data ground truth over all cv splits.  
            pimp = pimp[np.isnan(yimp)==0] # Remove NaNs (no ground truth)
            yimp = yimp[np.isnan(yimp)==0] # Remove NaNs (no ground truth)
            p[imp] = pimp[yimp.mask==False]#.compressed()
            y[imp] = yimp[yimp.mask==False]#.compressed()
        plot_roc_curves(y,p,ax=ax) # Plot on the given axes.  
        ax.set_title(list(Y_clean)[i]) # Set the title.  
    for i in range(i+1,len(axes.flat)):
        ax = axes.flat[i]
        ax.set_axis_off()
    plt.tight_layout()


def report(rs,props,imps):
    n_props = list(rs.values())[0].shape[0]
    assert n_props == len(props)
    n_splits = list(rs.values())[0].shape[1]
    for i,prop in enumerate(props):
        for imp in imps:
            vals = rs[imp][i,:]
            print('%s,%s: %.3f +/- %.3f' % (imp,prop,vals.mean(),vals.std()/np.sqrt(n_splits)))


def build_p_frame(p0,p1,pathological,guide):
    exclude = ['demunp','ftdpdefined','hs','cbdp'] # Exclude these labels from further analysis.  
    ps = [] # List of information for data frame.  
    for y,p_ in [(0,p0),(1,p1)]: # Iterate over no/yes and probabilities of no/yes.  
        for i in p_['all']: # Iterate over labels (AD, PD, etc.)
            label = pathological.columns.values[i] # Name of label.  
            if label in exclude: # Exclude if in the exclude list.  
                continue
            label = guide.query("Name=='%s'" % label)['Label'].values[0].replace('FD ','') # Fix label.  
            for value in p_['all'][i]: # Iterate over subjects.  
                ps.append((label,value,'+' if y else '--')) # Fill the list.  
    ps = pandas.DataFrame(ps, columns=['Diagnosis','Predicted Probability','Outcome']) # Convert to a data frame.  
    return ps


def fit_models(imps, X, Y, Y_clean, 
               labels=None, n_estimators=25, n_splits=5,):
    n_obs = X['missing'].shape[0] # Number of observations.  
    n_features = X['missing'].shape[1] # Number of observations.  
    n_props = Y['missing'].shape[1] # Number of properties to predict.  
    test_size = 0.2
    if labels is None:
        shuffle_split = ShuffleSplit(n_obs,n_iter=n_splits,
                                     test_size=test_size,random_state=0)
    else:
        shuffle_split = LabelShuffleSplit(labels,n_iter=n_splits,
                                          test_size=test_size,random_state=0)
    n_test_samples = np.max([len(list(shuffle_split)[i][1]) \
                            for i in range(n_splits)])
    rs = {imp:np.ma.zeros((n_props,n_splits)) for imp in imps}
    ps = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    ys = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    feature_importances = {imp:np.ma.zeros((n_props,n_features,n_splits)) for imp in imps}
    for prop in range(n_props):
        print("Fitting model for %s..." % list(Y_clean)[prop])
        for imp in imps:
            for i,(train,test) in enumerate(shuffle_split):
                X_train,X_test = X[imp][train],X[imp][test]
                Y_train,Y_test = Y[imp][train,prop],Y['missing'][test,prop]
                rfc = RandomForestClassifier(n_estimators=n_estimators,random_state=0)
                #if Y_train.shape[1] == 1:
                #    Y_train = Y_train.ravel()
                rfc.fit(X_train,Y_train)
                Y_predict = rfc.predict(X_test)#.reshape(-1,n_props)
                probs = rfc.predict_proba(X_test)
                if probs.shape[1]<2 and probs.mean()==1.0:
                    n_test_samples = len(probs)
                    ps[imp][prop,i,:n_test_samples] = 0.0
                else:
                    n_test_samples = len(probs[:,1])
                    ps[imp][prop,i,:n_test_samples] = probs[:,1]
                ys[imp][prop,i,:n_test_samples] = Y_test
                rs[imp][prop,i] = np.ma.corrcoef(Y_predict,Y_test)[0,1]
                feature_importances[imp][prop,:,i] = rfc.feature_importances_
    return rs,feature_importances,ys,ps


def imputation(clean,imps=['knn','nmm','softimpute','biscaler']):
    imputer = Imputer(strategy='median',axis=0)
    X = {'missing':clean.as_matrix().astype('float')}
    X['median'] = imputer.fit_transform(X['missing'])
    
    if 'knn' in imps:
        X['knn'] = KNN(k=3).complete(X['missing'])
    if 'nnm' in imps:
        X['nnm'] = NuclearNormMinimization().complete(X['missing'])
    if 'softimpute' in imps:
        X['softimpute'] = SoftImpute().complete(X['missing'])
    
    X['missing'] = np.ma.array(X['missing'],mask=np.isnan(X['missing']))
    return X


def display_importances(X_clean,Y_clean,feature_importances,style=''):
    diagnoses = [x.replace('Clinpath','').replace('Nos','NOS') \
                 for x in list(Y_clean)]
    df_importances = pandas.DataFrame(columns=diagnoses)
    f_importance_means = feature_importances['knn'].mean(axis=2)
    n_features = f_importance_means.shape[1]
    for i,diagnosis in enumerate(diagnoses):
        f_d_importance_means = [(feature[:20],f_importance_means[i,j].round(3)) for j,feature in enumerate(list(X_clean))]
        df_importances[diagnosis] = sorted(f_d_importance_means,key=lambda x:x[1],reverse=True)
    index = pandas.Index(range(1,n_features+1))
    df_importances.set_index(index,inplace=True)
    html = df_importances.head(10).to_html()
    bs = bs4.BeautifulSoup(html)
    for i,th in enumerate(bs.findAll('th')):
        th['width'] = '50px'
    for i,td in enumerate(bs.findAll('td')):
        feature,value = td.text.split(',')
        value = float(value.replace(')',''))
        feature = feature.replace('(','')
        size = 9+3*(value-0.02)/0.1
        td.string = feature.lower()
        td['style'] = 'font-size:%dpx;' % size
        if any([key in td.text for key in ['smell','upsit']]):
            td['style'] += 'color:rgb(255,0,0);'
    html = bs.html 
    #print(html)
    return HTML('<span style="%s">%s</span>' % (style,html))


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
