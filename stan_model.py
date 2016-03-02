from __future__ import division
import pystan
from upsit import bbdp,plt
import numpy as np
from scipy.stats import bernoulli
from datetime import datetime

code = """
    data {
        int<lower=0,upper=40> n_questions; // number of questions
        int<lower=0> n_subjects; // number of subjects
        int<lower=0> n_control; // number of subjects who are controls
        int<lower=0,upper=1> correct[n_subjects,n_questions]; // 1 if response is correct, 0 if not
        int<lower=0,upper=1> holdout[n_subjects,n_questions]; // Mask for responses; 1 to exclude, 0 to include.   
        real tau;
        real sigma;
        real sigma_pd;
    }
    parameters {
        real r_ctrl_mean; // Control group mean rating.  
        real r_pd_mean; // PD group mean ratimg.  
        real r_subject[n_subjects]; // Subject skill ratings.  
        real r_q[n_questions]; // Question difficulty ratings.  
        real r_q_pd[n_questions]; // Extra question difficulty penalty for PD.    
    }
    model {
        r_ctrl_mean ~ normal(0, tau);
        r_pd_mean ~ normal(0, tau);
        for (i in 1:n_control)
            r_subject[i] ~ normal(r_ctrl_mean, sigma);
        for (i in (n_control+1):n_subjects)
            r_subject[i] ~ normal(r_pd_mean, sigma);
        r_q ~ normal(0, 1);
        r_q_pd ~ normal(r_q, sigma_pd);
        for (i in 1:n_subjects) {
            for (q in 1:n_questions) {
                if( holdout[i][q] == 0) {
                    if (i <= n_control)
                        correct[i][q] ~ bernoulli_logit(r_subject[i]+r_q[q]);
                    else
                        correct[i][q] ~ bernoulli_logit(r_subject[i]+r_q_pd[q]);
                    }
                }
            }  
    }
    """

# Auxiliary functions.  
THEN = None
def tic():
    global THEN
    THEN = datetime.now()

def toc(activity='Something'):
    now = datetime.now()
    delta = now - THEN
    seconds = delta.days*24*3600 + delta.seconds + delta.microseconds/1e6
    print '%s took %.3g seconds' % (activity,seconds)

def logit(x):
    return 1.0 / (1.0+np.exp(-x))

def identity_permuted(size):
    return np.random.permutation(np.identity(size)).astype(int)

def get_posterior_means(fit):
    means = {}
    x = fit.extract()
    for key,value in x.items()[:-1]:
        means[key] = value.mean(axis=0)
    return means

# Data preparation.  
ctrl = {}
pd = {}
subjects,tests = bbdp.load()
scores = {}
for test in tests:
    if test.subject.label == 'ctrl':
        ctrl[test.subject] = []
        x = ctrl
    if test.subject.label == 'pd':
        pd[test.subject] = []
        x = pd
    for q in range(1,41):
        x[test.subject].append(int(test.response_set.responses[q].correct))
 
correct = np.array(ctrl.values() + pd.values(),dtype='int')
n_subjects,n_questions = correct.shape
n_fold = 25

holdout = identity_permuted(n_questions)
while(holdout.shape[0]<n_subjects):
    holdout = np.vstack((holdout,identity_permuted(n_questions)))
holdout = holdout[:n_subjects,:] # Truncate to n_subjects x n_questions.  

data = {'n_questions': n_questions,
        'n_subjects': n_subjects,
        'n_control' : len(ctrl),
        'correct': correct,
        'holdout': holdout,
        'tau': 1.0,
        'sigma': 0.6,
        'sigma_pd': 1.0}

# Main body.  
tic()
model = pystan.StanModel(model_code=code)
toc('Model compilation')

betas = 10.0**np.arange(-2,0.25,0.25)
cv_scores = []

for beta in betas:
    data['sigma2'] = beta
    log_ls_in = []
    log_ls_out = []

    for permutation in range(n_fold):
        print ("\n"*5 + "="*10 +"\nPermutation %d, Beta=%f\n" + "="*10 + "\n"*5) % (permutation,beta)
        data['holdout'] = np.random.permutation(holdout)
        
        tic()
        fit = model.sampling(data=data)
        toc('Model fitting')

        means = get_posterior_means(fit)
        
        log_l = 0
        for subject,question in zip(*np.where(holdout==0)):
            r_subject = means['r_subject'][subject]    
            r_q = means['r_q'][question]   
            if subject < len(ctrl):
                r_q = means['r_q'][question]
            if subject >= len(ctrl):
                r_q = means['r_q_pd'][question]
            is_correct = correct[subject][question]
            prob_correct = logit(r_subject + r_q)
            likelihood = bernoulli.pmf(is_correct,prob_correct)
            log_l += np.log(likelihood)
        num_not_held_out = len(np.where(holdout==0)[0])
        print "Log-likelihood per sample in sample is %.2f" % (log_l/num_not_held_out)
        log_ls_in.append(log_l/num_not_held_out)

        log_l = 0
        for subject,question in zip(*np.where(holdout!=0)):
            r_subject = means['r_subject'][subject]
            if subject < len(ctrl):
                r_q = means['r_q'][question]
            if subject >= len(ctrl):
                r_q = means['r_q_pd'][question]
            is_correct = correct[subject][question]
            prob_correct = logit(r_subject + r_q)
            likelihood = bernoulli.pmf(is_correct,prob_correct)
            log_l += np.log(likelihood)
        num_held_out = len(np.where(holdout!=0)[0])
        print "Log-likelihood per sample out of sample is %.2f" % (log_l/num_held_out)
        log_ls_out.append(log_l/num_held_out)

    cv_scores.append((log_ls_in,log_ls_out))

print betas
print cv_scores

plt.plot(betas,[np.mean(x[1]) for x in cv_scores])
plt.show()

