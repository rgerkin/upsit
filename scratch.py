import inspect
import builtins
import re
import time

import nbformat
from IPython.display import Image,display,HTML

import numpy as np
from scipy.special import beta as betaf 
from scipy.stats import norm,beta
from scipy.optimize import minimize
import seaborn as sns

import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score,LeaveOneOut,\
                                     ShuffleSplit,GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,\
                               OutputCodeClassifier
from sklearn.preprocessing import MultiLabelBinarizer,Imputer
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import seaborn as sns
import bs4

import bbdp
from upsit import plt

SAVE = False

class Prindent:
    def __init__(self):
        self.base_depth = len(self.get_stack())
        x = self.get_stack()
        #for xi in x:
        #    builtins.print(xi.filename)
    def get_stack(self):
        stack = [x for x in inspect.stack() if not \
                    any([y in x.filename for y in \
                        ['zmq','IPy','ipy','tornado','runpy',
                         'imp.py','importlib','traitlets']])]
        return stack
    def print(self,*args,add=0,**kwargs):
        stack = self.get_stack()
        #builtins.print(len(stack),self.base_depth)
        depth = len(stack)-self.base_depth+add
        args = ["\t"*depth + "%s"%arg for arg in args]
        #builtins.print(len(inspect.stack()),self.base_depth)
        builtins.print(*args,**kwargs)


print = Prindent().print


def save_fig():
    global SAVE
    if SAVE:
        plt.savefig('%s.png' % time.time(), format='png', dpi=600)


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
        if 'demented' in options:
            try:
                x += [test.subject.demented]
            except:
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


def plot_cumul(X,Y,label):
    X_pos = X[Y == True]
    X_neg = X[Y == False]
    plt.plot(sorted(X_neg),np.linspace(0,1,len(X_neg)),'k',label='-')
    plt.plot(sorted(X_pos),np.linspace(0,1,len(X_pos)),'r',label='+')
    plt.xlabel(label)
    plt.ylabel('Cumulative Probability')
    plt.ylim(0,1)
    plt.legend(loc=2)


def cross_validate(clf,X,Y,cv,kind):
    mean = cross_val_score(clf,X,Y,cv=cv).mean()
    print("Cross-validation accuracy for %s is %.3f" % (kind,mean))


def makeXY(keys,uses,x,drop_more=[],restrict=[]):
    X = {key:{} for key in keys}
    Y = {}
    for key in keys:
        pos,neg = [_.strip() for _ in key.split('vs')]
        if restrict: # Not implmented.  
            for kind in pos,neg:
                pass#x[kind].drop(inplace=1)
        n = len(x[pos])+len(x[neg])
        #order = np.random.permutation(range(n))
        for use,regex in uses.items():
            drop = []
            for reg in regex:
                drop += [feature for feature in list(x[pos]) if re.match(reg,feature)]
            drop = list(set(drop))
            drop += drop_more
            x_new = pd.concat([x[_].drop(drop,axis=1) for _ in (pos,neg)])
            #x_new = normalize(x_new,norm='max',axis=0)
            #from sklearn.decomposition import PCA,NMF
            #pca = PCA(n_components = min(10,x_new.shape[1]))
            #nmf = NMF(n_components = min(5,x_new.shape[1]))
            #x_new = pca.fit_transform(x_new)
            #x_new = nmf.fit_transform(x_new)
            X[key][use] = x_new#[order,:]
        Y[key] = pd.Series(index=x_new.index,data=np.ones(n))
        Y[key].loc[x[neg].index] = 0
    return X,Y


def make_py(keys,uses,X,Y,clfs,ignore=None):
    p_pos = {}
    ys = {}
    for key in keys:
        #print(key)
        splitter = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
        p_pos[key] = {}
        for use in uses:
            #print("\t%s" % use)
            p_pos[key][use] = {'average':0}
            #x = 0
            for clf in clfs:
                p,ys[key] = get_ps(clf,splitter,X[key][use],Y[key],ignore=ignore); # Extract the probability of PD from each classifier
                p = p.clip(1e-15,1-1e-15)
                p_pos[key][use][str(clf)[:7]] = p
                p_pos[key][use]['average'] += p
            p_pos[key][use]['average'] /= len(clfs)
    return p_pos,ys


def plot_rocs(keys,uses,p_pos,Y,ys,smooth=True,no_plot=False):
    sns.set(font_scale=2)
    aucs = {}
    aucs_sd = {}
    for key in keys:
        if not no_plot:
            plt.figure()
        d = {use:p_pos[key][use]['average'] for use in uses}
        n0 = sum(Y[key]==0)
        n1 = sum(Y[key]>0)
        if n0==0 or n1==0:
            print("Missing either positive or negative examples of %s" % key)
            continue
        if ys[key].std()==0:
            print("Missing either positive or negative bootstrap examples of %s" % key)
            continue
        aucs[key],aucs_sd[key] = plot_roc_curve(ys[key],n0=n0,n1=n1,smooth=smooth,no_plot=no_plot,**d)
        for i,(a1,sd1) in enumerate(zip(aucs[key],aucs_sd[key])):
            for j,(a2,sd2) in enumerate(zip(aucs[key],aucs_sd[key])):
                if i>j:
                    d = np.abs((a1-a2)/np.sqrt((sd1**2 + sd2**2)/2))
                    p = (1-norm.cdf(d,0,1))/2
                    print("\t%s vs %s: p=%.4f" % (sorted(uses)[i],sorted(uses)[j],p))
        if not no_plot:
            plt.title(keys[key])
        save_fig()
    return aucs,aucs_sd


def get_ps(clf,splitter,X,Y,ignore=None):
    ps = []
    ys = []
    assert np.array_equal(X.index,Y.index),"X and Y indices must match"
    for i, (train, test) in enumerate(splitter.split(Y)):
        train = X.index[train]
        test = X.index[test]
        try:
            assert Y.loc[train].mean() not in [0.0,1.0], \
            "Must have both positive and negative examples"
        except AssertionError as e:
            print("Skipping split %d because: %s" % (i,e))
        else:
            clf.fit(X.loc[train], Y.loc[train])
            X_test = X.loc[test].drop(ignore,errors='ignore') if ignore else X.loc[test]
            Y_test = Y.loc[test].drop(ignore,errors='ignore') if ignore else Y.loc[test]
            n_test_samples = X_test.shape[0]
            if n_test_samples:
                ps += list(clf.predict_proba(X_test)[:,1])
                ys += list(Y_test)
            else:
                print("Skipping split %d because there are no test samples" % i)
    return np.array(ps),np.array(ys)


def get_roc_curve(Y,p,smooth=False):
    if not smooth:
        fpr, tpr, thresholds = roc_curve(Y, p)
    else:
        from scipy.stats import gaussian_kde
        x = -norm.isf(np.array(p))
        x0 = x[Y==0]
        x1 = x[Y==1]
        threshold = np.linspace(-10,10,201)
        fpr = [gaussian_kde(x0,0.2).integrate_box(t,np.inf) for t in threshold]
        tpr = [gaussian_kde(x1,0.2).integrate_box(t,np.inf) for t in threshold]
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr = 1-np.array(fpr)
        tpr = 1-np.array(tpr)
        roc_auc = 1-roc_auc
    return fpr,tpr,roc_auc


def binormal_roc(Y,p):
    x = -norm.isf(np.array(p))
    mu0 = x[Y==0].mean()
    sigma0 = x[Y==0].std()
    mu1 = x[Y==1].mean()
    sigma1 = x[Y==1].std()
    # Separation
    a = (mu1-mu0)/sigma1
    # Symmetry
    b = sigma0/sigma1
    threshold = np.linspace(0,1,1000)
    roc = norm.cdf(a-b*norm.isf(threshold))
    return threshold,roc


def bibeta_roc(Y,p):
    def logL(ab):
        a0,b0,a1,b1 = ab
        LL = beta.logpdf(p[Y==0],a0,b0).sum() + beta.logpdf(p[Y==1],a1,b1).sum() 
        return -LL
    result = minimize(logL,[1,3,3,1],bounds=[(1e-7,None)]*4)
    a0,b0,a1,b1 = result.x
    threshold = np.linspace(0,1,1000)
    fpr = 1-beta.cdf(threshold,a0,b0)
    tpr = 1-beta.cdf(threshold,a1,b1)
    return threshold,fpr,tpr


def rgb2hex(r,g,b):
    return "#%0.2x%0.2x%0.2x" % (r,g,b)


def get_colors(i):
    black = rgb2hex(0, 0, 0)
    blue = rgb2hex(0, 0, 255)
    red = rgb2hex(255, 0, 0)
    green = rgb2hex(0, 255, 0)
    magenta = rgb2hex(255, 0, 255)
    brown = rgb2hex(128, 0, 0)
    yellow = rgb2hex(255, 255, 0)
    pink = rgb2hex(255, 128, 128)
    gray = rgb2hex(128, 128, 128)
    orange = rgb2hex(255, 128, 0)

    colors = [black,blue,red,green,magenta,brown,yellow,pink,gray,orange]
    return colors[i % len(colors)]


def plot_roc_curve(Y,n0=None,n1=None,smooth=False,no_plot=False,**ps):
    aucs = []
    aucs_sd = []
    if n0 is None:
        n0 = sum(Y==0)
    if n1 is None:
        n1 = sum(Y>0)
    for i,(title,p) in enumerate(sorted(ps.items())):
        fpr,tpr,auc = get_roc_curve(Y,p,smooth=smooth)
        aucs.append(auc)
        # Confidence Intervals for the Area under the ROC Curve
        # Cortes and Mohri
        # http://www.cs.nyu.edu/~mohri/pub/area.pdf
        m = n1
        n = n0
        A = auc
        Pxxy = 0
        Pxyy = 0
        iters = 10000
        for j in range(iters):
            index = np.arange(len(Y))
            np.random.shuffle(index)
            p_shuff = p[index]
            Y_shuff = Y[index]
            pa,pb = p_shuff[Y_shuff>0][0:2]
            na,nb = p_shuff[Y_shuff==0][0:2]
            Pxxy += ((pa>na) and (pb>na))
            Pxyy += ((na<pa) and (nb<pa))
        Pxxy/=iters
        Pxyy/=iters
        #print(A,Pxxy,Pxyy,m,n)
        var = (A*(1-A)+(m-1)*(Pxxy-(A**2))+(n-1)*(Pxyy-(A**2)))/(m*n)
        sd = np.sqrt(var)
        aucs_sd.append(sd)
        if not no_plot:
            plt.plot(fpr, tpr, lw=2, color=get_colors(i), label='%s = %0.2f' % (title,auc))
        else:
            print('%s = %0.3f +/- %0.3f' % (title,auc,sd))
    if not no_plot:
        plt.xlabel('False Positive Rate')#, fontsize='large', fontweight='bold')
        plt.ylabel('True Positive Rate')#, fontsize='large', fontweight='bold')
        plt.title('ROC curves')#, fontsize='large', fontweight='bold')
        plt.xticks()#fontsize='large', fontweight='bold')
        plt.yticks()#fontsize='large', fontweight='bold')
        plt.xlim(-0.01,1.01)
        plt.ylim(-0.01,1.01)
        plt.legend(loc="lower right",fontsize=17)
    return aucs,aucs_sd


def plot_roc_curves(Y,p,ax=None,label='full',title='AUC',color=None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    colors = {'basic':'gray','total':'pink','all':'red'}
    aucs = {}
    for key in p:
        fpr,tpr,auc = get_roc_curve(Y[key],p[key])
        if color is None:
            color = 'red' if key not in colors else colors[key]
        ax.plot(fpr, tpr, lw=2, color=color, 
                label={'full':'AUC using\n%s = %0.2f' % (key,auc),
                       'sparse':'%s = %.2f' % (title,auc)}[label])
        aucs[key] = auc
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
    return aucs


def roc_data(X,Y,clf,n_iter=50,test_size=0.1):
    if n_iter is None and test_size is None:
        cv = LeaveOneOut()
    else:
        cv = ShuffleSplit(n_iter=n_iter,test_size=test_size)
    n_labels = Y.shape[1]
    Y_cv = {i:[] for i in range(n_labels)}
    p = {i:[] for i in range(n_labels)}
    p_1 = {i:[] for i in range(n_labels)}
    p_0 = {i:[] for i in range(n_labels)}
    for train, test in cv.split(Y):
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


def plot_just_rocs(props,ps,ys,imps,plot=True,axes=None,m=None,n=None,color='k',title='AUC'):
    if plot:
        if axes is None:
            if m is None:
                m = max(1,1+int(len(props)/4))
            if n is None:
                n = min(4,len(props))
            fig,axes = plt.subplots(m,n,sharex=True,sharey=True,
                                    squeeze=False,figsize=(m*3.5,n*3.5))
    y = {}
    p = {}
    aucs = {}
    for i,prop in enumerate(props):
        if plot:
            ax = axes.flat[i]
        for imp in imps:
            pimp = ps[imp][i,:,:].ravel() # Unravel all predictions of test data over all cv splits.  
            yimp = ys[imp][i,:,:].ravel() # Unravel all test data ground truth over all cv splits.  
            pimp = pimp[np.isnan(yimp)==0] # Remove NaNs (no ground truth)
            yimp = yimp[np.isnan(yimp)==0] # Remove NaNs (no ground truth)
            p[imp] = pimp[yimp.mask==False]#.compressed()
            y[imp] = yimp[yimp.mask==False]#.compressed()
        if plot:
            aucs[prop] = plot_roc_curves(y,p,ax=ax,label='sparse',title=title,color=color) # Plot on the given axes.  
        else:
            aucsi = {}
            for imp in p:
                fpr,tpr,auc = get_roc_curve(y[imp],p[imp])
                aucsi[imp] = auc
            aucs[prop] = aucsi
        if plot:
            ax.set_title(props[i].replace('Clinpath ','').replace('Nos','NOS')) # Set the title.  
    if plot:
        for i in range(i+1,len(axes.flat)):
            ax = axes.flat[i]
            ax.set_axis_off()
        plt.tight_layout()
    return aucs,axes


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


def fit_models(imps, X, Y, all_props, props=None,
               labels=None, n_splits=5, 
               clf_args={'n_estimators':25, 
                         'max_features':'auto', 
                         'random_state':0}):
    if props is None:
        props = all_props
    n_obs = X['missing'].shape[0] # Number of observations.  
    n_features = X['missing'].shape[1] # Number of observations.  
    n_props = len(props) # Number of properties to predict.  
    test_size = 0.2
    if labels is None:
        shuffle_split = ShuffleSplit(n_iter=n_splits,
                                     test_size=test_size,random_state=0)
    else:
        shuffle_split = GroupShuffleSplit(n_iter=n_splits,
                                          test_size=test_size,random_state=0)
    n_test_samples = np.max([len(list(shuffle_split.split(range(n_obs),groups=labels))[i][1]) \
                            for i in range(n_splits)])
    rs = {imp:np.ma.zeros((n_props,n_splits)) for imp in imps}
    ps = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    ys = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    feature_importances = {imp:np.ma.zeros((n_props,n_features,n_splits)) for imp in imps}
    for n_prop,prop in enumerate(props):
        j = all_props.index(prop)
        print("Fitting model for %s..." % prop)
        for imp in imps:
            for k,(train,test) in enumerate(shuffle_split.split(range(n_obs),
                                                                groups=labels)):
                X_train,X_test = X[imp][train],X[imp][test]
                Y_train,Y_test = Y[imp][train,j],Y['missing'][test,j]
                clf_args_ = {key:(value if type(value) is not dict \
                             else value[prop])\
                             for key,value in clf_args.items()}
                if clf_args_['max_features'] not in [None, 'auto']:
                   clf_args_['max_features'] = min(X_train.shape[1],
                                                   clf_args_['max_features'])
                rfc = RandomForestClassifier(**clf_args_)
                #if Y_train.shape[1] == 1:
                #    Y_train = Y_train.ravel()
                rfc.fit(X_train,Y_train)
                Y_predict = rfc.predict(X_test)#.reshape(-1,n_props)
                probs = rfc.predict_proba(X_test)
                if probs.shape[1]<2 and probs.mean()==1.0:
                    n_test_samples = len(probs)
                    ps[imp][n_prop,k,:n_test_samples] = 0.0
                else:
                    n_test_samples = len(probs[:,1])
                    ps[imp][n_prop,k,:n_test_samples] = probs[:,1]
                ys[imp][n_prop,k,:n_test_samples] = Y_test
                rs[imp][n_prop,k] = np.ma.corrcoef(Y_predict,Y_test)[0,1]
                feature_importances[imp][n_prop,:,k] = rfc.feature_importances_
    return rs,feature_importances,ys,ps


def fit_models_mc(imps, X, Y, all_props, props=None,
               labels=None, n_splits=5, 
               clf_args={'n_estimators':25, 
                         'max_features':'auto', 
                         'random_state':0}):
    if props is None:
        props = all_props
    n_obs = X['missing'].shape[0] # Number of observations.  
    n_features = X['missing'].shape[1] # Number of observations.  
    n_props = len(props) # Number of properties to predict.  
    test_size = 0.2
    if labels is None:
        shuffle_split = ShuffleSplit(n_iter=n_splits,
                                     test_size=test_size,random_state=0)
    else:
        shuffle_split = LabelShuffleSplit(n_iter=n_splits,
                                          test_size=test_size,random_state=0)
    n_test_samples = np.max([len(list(shuffle_split)[i][1]) \
                            for i in range(n_splits)])
    rs = {imp:np.ma.zeros((n_props,n_splits)) for imp in imps}
    ps = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    ys = {imp:np.ma.masked_all((n_props,n_splits,n_test_samples)) for imp in imps}
    feature_importances = None#{imp:np.ma.zeros((n_props,n_features,n_splits)) for imp in imps}
    cols = np.array([i for i in range(len(all_props)) if all_props[i] in props])
    for imp in imps:
        for k,(train,test) in enumerate(shuffle_split.split(range(n_obs),groups=labels)):
            #X_train,X_test = X[imp][train][:,cols],X[imp][test][:,cols]
            #Y_train,Y_test = Y[imp][train][:,cols],Y['missing'][test][:,cols]
            X_train,X_test = X[imp][train,:],X[imp][test,:]
            Y_train,Y_test = Y[imp][train,:],Y['missing'][test,:]
            clf_args_ = {key:(value if type(value) is not dict \
                         else value[prop])\
                         for key,value in clf_args.items()}
            if clf_args_['max_features'] not in [None, 'auto']:
               clf_args_['max_features'] = min(X_train.shape[1],
                                               clf_args_['max_features'])
            rfc = RandomForestClassifier(**clf_args_)
            onevsrest = OneVsRestClassifier(rfc)
            onevsrest.fit(X_train,Y_train)
            Y_predict = onevsrest.predict(X_test)#.reshape(-1,n_props)
            probs = onevsrest.predict_proba(X_test)
            if probs.shape[1]<2 and probs.mean()==1.0:
                n_test_samples = len(probs)
                ps[imp][:,k,:n_test_samples] = 0.0
            else:
                n_test_samples = len(probs[:,1])
                ps[imp][:,k,:n_test_samples] = probs.T
            ys[imp][:,k,:n_test_samples] = Y_test.T
            for i in range(n_props):
                rs[imp][i,k] = np.ma.corrcoef(Y_predict[:,i],Y_test[:,i])[0,1]
            #feature_importances[imp][n_prop,:,k] = onevsrest.feature_importances_
    return rs,feature_importances,ys,ps


def scatter_diag(props,ps,os,x_diag,y_diag,plot=True):
    from matplotlib.colors import Colormap as cmap
    imp = 'knn'
    xi = props.index(x_diag)
    yi = props.index(y_diag)
    p_x = ps[imp][xi,:,:].ravel() # Unravel all predictions of test data over all cv splits.  
    p_y = ps[imp][yi,:,:].ravel() # Unravel all test data ground truth over all cv splits.  
    o_x = os[imp][xi,:,:].ravel() # Unravel all predictions of test data over all cv splits.  
    o_y = os[imp][yi,:,:].ravel() # Unravel all test data ground truth over all cv splits.  
    mask = o_x.mask + o_y.mask
    p_x = p_x[mask==False]
    p_y = p_y[mask==False]
    o_x = o_x[mask==False]
    o_y = o_y[mask==False]
    colors = np.vstack((o_x.data,np.zeros(len(o_x)),o_y.data)).T
    colors[colors==0] = 0.2
    if plot:
        plt.figure(figsize=(10,10))
        plt.scatter(p_x+0.02*np.random.rand(p_pd.shape[0]),
                    p_y+0.02*np.random.rand(p_pd.shape[0]),
                    s=15,
                    c=colors)
        plt.xlabel(x_diag)
        plt.ylabel(y_diag)
        plt.xlim(0,p_x.max()*1.05)
        plt.ylim(0,p_y.max()*1.05)
        plt.legend()
    return p_x,p_y,o_x,o_y


def roc_showdown(p_x,p_y,o_x,o_y,x_diag,y_diag,title='AUC',color='black'):
    from sklearn.metrics import roc_curve,auc
    p = p_x - p_y
    o = o_x - o_y
    p = p[np.abs(o)==1] # Only cases where x or y equals 1, but not both.  
    o = o[np.abs(o)==1]
    o = o==1
    fpr,tpr,_ = roc_curve(o, p)
    plt.plot(fpr,1-tpr,label="%s = %.3f" % (title,auc(fpr,tpr)),c=color)
    x_diag = x_diag.replace('Clinpath ','').replace('Nos','NOS')
    y_diag = y_diag.replace('Clinpath ','').replace('Nos','NOS')
    plt.xlabel('False %s rate' % x_diag)#'Fraction %s misdiagnosed as %s' % (y_diag,x_diag))
    plt.ylabel('False %s rate' % y_diag)#'Fraction %s misdiagnosed as %s' % (x_diag,y_diag))
    #plt.legend(loc=1)
    

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


def display_importances(all_features,props,feature_importances,style=''):
    props = [x.replace('Clinpath','').replace('Nos','NOS') \
                 for x in props]
    df_importances = pandas.DataFrame(columns=props)
    f_importance_means = feature_importances['knn'].mean(axis=2)
    n_features = f_importance_means.shape[1]
    for i,prop in enumerate(props):
        f_d_importance_means = [(feature[:20],f_importance_means[i,j].round(3)) for j,feature in enumerate(all_features)]
        df_importances[prop] = sorted(f_d_importance_means,key=lambda x:x[1],reverse=True)
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
        td.string = feature.lower()#+'(%.3f)'%value
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
