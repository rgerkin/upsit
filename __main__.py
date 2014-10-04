from __future__ import division
from pprint import pprint

import numpy as np

import upsit
import ppmi
import bbdp

if __name__ == '__main__':
    """PPMI data"""
    '''
    ppmi_data = ppmi.load()
    ppmi.plot_cumul_hist(ppmi_data,booklet=1)
    ppmi.plot_cumul_hist(ppmi_data,booklet=2)
    ppmi.plot_cumul_hist(ppmi_data,booklet=3)
    ppmi.plot_cumul_hist(ppmi_data,booklet=4)
    upsit.plt.show()
    '''
    
    """BBDP data"""
    subjects,tests = bbdp.load()
    ctrl = []; pd = []
    
    scores = {}
    ctrl = []
    pd = []
    if 1:
        for q in range(1,41):
            scores[q] = {'ctrl':[],'pd':[]}
            for test in tests:
                correct = test.response_set.responses[q].correct
                if test.is_control():
                    scores[q]['ctrl'].append(correct)
                else:
                    scores[q]['pd'].append(correct)    
            def print_scores(kind):
                print '%s: %d/%d' % (kind,sum(scores[q][kind]),len(scores[q][kind]))
            #print "\r\rQuestion #%d:" % q
            #print_scores('ctrl')
            #print_scores('pd')
            def proportion(i,kind):
                return sum(scores[i][kind])/len(scores[i][kind])
            ctrl.append(proportion(q,'ctrl'))
            pd.append(proportion(q,'pd'))
            #upsit.plt.scatter(np.array(ctrl[q-1]),np.array(pd[q-1]),
            #                  marker='${}$'.format(q),
            #                  s=96, 
            #                  color='black')
        
        #print np.array(ctrl),np.array(pd)
        #upsit.plt.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),'k')
        
        diff = np.zeros(40)
        for q in range(1,41):
            ctrl_p = proportion(q,'ctrl')
            pd_p0 = proportion(q,'pd')
            pd_p = np.random.binomial(23,pd_p0) / 23.0
            diff[q-1] =  ctrl_p - pd_p

        shuffle_diff = np.zeros(40000)
        for q in range(1,40001):
            q_ctrl = ((q-1) % 40) + 1
            q_pd = np.random.randint(1,41)
            ctrl_p = proportion(q_ctrl,'ctrl')
            ctrl_x = np.log(ctrl_p/(1-ctrl_p))
            pd_x = ctrl_x - 1.4 + 0.7*np.random.randn()
            pd_p0 = np.exp(pd_x)/(1+np.exp(pd_x))
            pd_p = np.random.binomial(23,pd_p0) / 23.0
            shuffle_diff[q-1] = ctrl_p - pd_p
        
        upsit.plt.figure()
        upsit.cumul_hist(diff,color='r')
        upsit.cumul_hist(shuffle_diff,color='k')

        upsit.plt.show()

    if 0:
        for test in tests:
    	   if test.subject.label == 'ctrl':
    		  ctrl.append(test.score)
    	   if test.subject.label == 'pd':
    		  pd.append(test.score)
    
        #pprint(upsit_key)
        #pprint(bbdp_data)
        #upsit.cumul_hist(ctrl,color='k')
        #upsit.cumul_hist(pd,color='r')
        #upsit.plt.show()
    