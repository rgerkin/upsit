from pprint import pprint

import upsit
import ppmi
import bbdp

if __name__ == '__main__':
    """PDBP data"""
    ppmi_data = ppmi.load()
    subjects,tests = bbdp.load()
    #pprint(upsit_key)
    #pprint(bbdp_data)
    '''pdbp.plot_cumul_hist(pdbp_data,booklet=1)
    pdbp.plot_cumul_hist(pdbp_data,booklet=2)
    pdbp.plot_cumul_hist(pdbp_data,booklet=3)
    pdbp.plot_cumul_hist(pdbp_data,booklet=4)
    upsit.plt.show()
    '''