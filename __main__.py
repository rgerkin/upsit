import upsit
import pdbp
import bbdp

if __name__ == '__main__':
    """PDBP data"""
    pdbp_data = pdbp.load()
    pdbp.plot_cumul_hist(pdbp_data,booklet=1)
    pdbp.plot_cumul_hist(pdbp_data,booklet=2)
    pdbp.plot_cumul_hist(pdbp_data,booklet=3)
    pdbp.plot_cumul_hist(pdbp_data,booklet=4)
    upsit.plt.show()