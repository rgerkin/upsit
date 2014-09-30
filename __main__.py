from pprint import pprint

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
    