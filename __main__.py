import upsit
import pdbp
import bbdp

if __name__ == '__main__':
    """PDBP data"""
    pdbp_data = pdbp.load()
    pdbp.plot_cumul_hist(pdbp_data)
    upsit.plt.show()