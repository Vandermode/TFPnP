from tfpnp.utils.visualize import seq_plot
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = [1,4,5,6,7,8,9,8,6,5,9,10,11]
    seq_plot(a, 'x', 'y')
    plt.savefig('foo.png')