import os
import matplotlib.pyplot as plt


save_path = os.getcwd()

def save_fig(save_path=save_path, figname='', dpi=1000, pdf=False, show=False):
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, figname + '.png'),
                dpi=dpi,
                transparent=True)
    if pdf:
        plt.savefig(os.path.join(save_path, figname + '.pdf'), transparent=True)
    if show:
        plt.show()
    print('Figure saved at: {}'.format(os.path.join(save_path, figname)))
    plt.close()