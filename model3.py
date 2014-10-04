import pystan
from upsit import bbdp,plt
import numpy as np

code = """
    data {
        int<lower=0,upper=40> n_questions; // number of questions
        int<lower=0> n_ctrl; // number of control subjects
        int<lower=0> n_pd; // number of pd subjects
        
        //int<lower=0,upper=40> score_ctrl[n_ctrl]; // UPSIT scores for controls
        //int<lower=0,upper=40> score_pd[n_pd]; // UPSIT scores for pd subjects
        
        int<lower=0,upper=1> correct_ctrl[n_ctrl,n_questions];
        int<lower=0,upper=1> correct_pd[n_pd,n_questions];

    }
    parameters {
        real r_ctrl_mean; // Control group mean rating.  
        real r_pd_mean; // PD group mean ratimg.  
        real r_ctrl[n_ctrl]; // Control subject ratings.  
        real r_pd[n_pd]; // PD subject ratings.  
        real r_q[n_questions]; // Question difficulty ratings.  
    }
    transformed parameters {
        real<lower=0,upper=1> p_ctrl[n_ctrl,n_questions];
        real<lower=0,upper=1> p_pd[n_pd,n_questions];

        for (q in 1:n_questions) {
            for (j in 1:n_ctrl) {
                p_ctrl[j][q] <- exp(r_ctrl[j]-r_q[q])/(1+exp(r_ctrl[j]-r_q[q]));
                }
            for (j in 1:n_pd) {
                p_pd[j][q] <- exp(r_pd[j]-r_q[q])/(1+exp(r_pd[j]-r_q[q]));
                }
            }
    }
    model {
        r_ctrl_mean ~ normal(0, 1);
        r_pd_mean ~ normal(0, 1);
        r_ctrl ~ normal(r_ctrl_mean, 1);
        r_pd ~ normal(r_pd_mean, 1);
        r_q ~ normal(0, 1);
        for (q in 1:n_questions) {
            for (j in 1:n_ctrl) {
                correct_ctrl[j][q] ~ binomial(1, p_ctrl[j][q]);
                }
            for (j in 1:n_pd) {
                correct_pd[j][q] ~ binomial(1, p_pd[j][q]); 
                }
            }  
    }
    """

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
 
data = {'n_questions': 40,
        'n_ctrl': len(ctrl),
        'n_pd' : len(pd),
        'correct_ctrl': np.array(ctrl.values()),
        'correct_pd': np.array(pd.values())}

model = pystan.StanModel(model_code=code)
fit = model.sampling(data=data)
print(fit)
fit.get_posterior_mean()
