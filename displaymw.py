import numpy as np
from pymoo.problems.multi.mw import *
import matplotlib.cm as cm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class DisplayMW1(MW1):

    def __init__(self, n_var=15):
        super(DisplayMW1, self).__init__(n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1.5, 400))
        z = np.ones_like(x)
        region = x + y - 1 - 0.5 * np.power(np.sin(2 * np.pi * (np.sqrt(2) * y - np.sqrt(2) * x)), 8)
        feasible_region = (region <= 0)
        z[np.logical_and(feasible_region, 0.85 * x + y >= 1)] = 0
        return x, y, z


class DisplayMW2(MW2):

    def __init__(self, n_var=15, **kwargs):
        super(DisplayMW2, self).__init__(n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1.5, 400))
        z = np.ones_like(x)
        region = x + y - 1 - 0.5 * np.power(np.sin(3 * np.pi * (np.sqrt(2) * y - np.sqrt(2) * x)), 8)
        feasible_region = (region <= 0)
        z[np.logical_and(feasible_region, x + y >= 1)] = 0
        return x, y, z


class DisplayMW3(MW3):

    def __init__(self, n_var=15):
        super(DisplayMW3, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1, 2000), np.linspace(0, 1.5, 2000))
        z = np.ones_like(x)
        feasible1 = (x + y - 1.05 - 0.45 * np.power(np.sin(0.75 * np.pi * (np.sqrt(2) * y - np.sqrt(2) * x)), 6) <= 0)
        feasible2 = (0.85 - x - y + 0.3 * np.power(np.sin(0.75 * np.pi * (np.sqrt(2) * y - np.sqrt(2) * x)), 2) <= 0)
        fes = np.logical_and(feasible1, feasible2)
        z[np.logical_and(fes, x + y >= 1)] = 0
        return x, y, z


class DisplayMW4(MW4):

    def __init__(self, n_var=None, n_obj=3):
        super(DisplayMW4, self).__init__(n_var=n_var, n_obj=n_obj)

    def get_pf_region(self):
        if self.n_obj == 2:
            x, y = np.meshgrid(np.linspace(0, 1.2, 400), np.linspace(0, 1.2, 400))
            z = np.ones_like(x)
            feasible = (x + y - 1 - 0.4 * np.power(np.sin(2.5 * np.pi * (y - x)), 8) <= 0)
            z[np.logical_and(feasible, x + y >= 1)] = 0
            return x, y, z

        elif self.n_obj == 3:
            a = np.linspace(0, 1, 10)[np.newaxis, :].T
            x = a * a.T
            y = a * (1 - a.T)
            z = (1 - a) * np.ones_like(a.T)
            return x, y, z

        else:
            z = np.array([])
            return z


class DisplayMW5(MW5):

    def __init__(self, n_var=15):
        super(DisplayMW5, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1.7, 400), np.linspace(0, 1.7, 400))
        z = np.ones_like(x)
        l1 = np.arctan(y / x)
        l2 = 0.5 * np.pi - 2 * np.abs(l1 - 0.25 * np.pi)
        fes1 = (x ** 2 + y ** 2 - (1.7 - 0.2 * np.sin(2 * l1)) ** 2 <= 0)
        fes2 = ((1 + 0.5 * np.sin(6 * np.power(l2, 3))) ** 2 - x ** 2 - y ** 2 <= 0)
        fes3 = ((1 - 0.45 * np.sin(6 * np.power(l2, 3))) ** 2 - x ** 2 - y ** 2 <= 0)
        temp = np.logical_and(np.logical_and(fes1, fes2), fes3)
        z[np.logical_and(temp, (x ** 2 + y ** 2 >= 1))] = 0
        return x, y, z


class DisplayMW6(MW6):

    def __init__(self, n_var=15):
        super(DisplayMW6, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1.7, 400), np.linspace(0, 1.7, 400))
        z = np.ones_like(x)
        l = np.cos(6 * np.power(np.arctan(y / x), 4)) ** 10
        fes = ((x / (1 + 0.15 * l)) ** 2 + (y / (1 + 0.75 * l)) ** 2 - 1 <= 0)
        z[np.logical_and(fes, x ** 2 + y ** 2 >= 1.21)] = 0
        return x, y, z


class DisplayMW7(MW7):

    def __init__(self, n_var=15):
        super(DisplayMW7, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1.5, 400), np.linspace(0, 1.5, 400))
        z = np.ones_like(x)
        l = np.arctan(y / x)
        fes1 = x ** 2 + y ** 2 - (1.2 + 0.4 * np.sin(4 * l) ** 16) ** 2 <= 0
        fes2 = (1.15 - 0.2 * np.sin(4 * l) ** 8) ** 2 - x ** 2 - y ** 2 <= 0
        z[fes1 & fes2 & (x ** 2 + y ** 2 >= 1)] = 0
        return x, y, z


class DisplayMW8(MW8):

    def __init__(self, n_var=None, n_obj=3):
        super(DisplayMW8, self).__init__(n_var=n_var, n_obj=n_obj)

    def get_pf_region(self):
        if self.n_obj == 2:
            x, y = np.meshgrid(np.linspace(0, 1.2, 400), np.linspace(0, 1.2, 400))
            z = np.ones_like(x)
            fes = x ** 2 + y ** 2 - (1.25 - 0.5 * np.sin(6 * np.arcsin(y / np.sqrt(x ** 2 + y ** 2))) ** 2) ** 2 <= 0
            z[fes & (x ** 2 + y ** 2 >= 1)] = 0
            return x, y, z

        elif self.n_obj == 3:
            a = np.linspace(0, np.pi / 2, 40)[np.newaxis, :].T
            x = np.sin(a) * np.cos(a.T)
            y = np.sin(a) * np.sin(a.T)
            z = np.cos(a) * np.ones_like(a.T)
            fes = 1 - (1.25 - 0.5 * np.sin(6 * np.arcsin(z)) ** 2) ** 2 <= 0
            z = z[fes]
            return x, y, z


class DisplayMW9(MW9):

    def __init__(self, n_var=15):
        super(DisplayMW9, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1.7, 400), np.linspace(0, 1.7, 400))
        z = np.ones_like(x)
        t1 = (1 - 0.64 * (x ** 2) - y) * (1 - 0.36 * (x ** 2) - y)
        t2 = 1.35 ** 2 - (x + 0.35) ** 2 - y
        t3 = 1.15 ** 2 - (x + 0.15) ** 2 - y
        # fes = np.min(t1, t2 * t3) <= 0
        g_index = (t1 >= t2 * t3)
        t1[g_index] = (t2 * t3)[g_index]
        fes = (t1 <= 0)
        z[fes & (np.power(x, 0.6) + y >= 1)] = 0
        return x, y, z


class DisplayMW10(MW10):

    def __init__(self, n_var=15):
        super(DisplayMW10, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1.5, 400))
        z = np.ones_like(x)
        fes1 = -1 * (2 - 4 * (x ** 2) - y) * (2 - 8 * (x ** 2) - y) <= 0
        fes2 = (2 - 2 * (x ** 2) - y) * (2 - 16 * (x ** 2) - y) <= 0
        fes3 = (1 - x ** 2 - y) * (1.2 - 1.2 * (x ** 2) - y) <= 0
        z[fes1 & fes2 & fes3 & (x ** 2 + y >= 1)] = 0
        return x, y, z


class DisplayMW11(MW11):

    def __init__(self, n_var=15):
        super(DisplayMW11, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 2.1, 400), np.linspace(0, 2.1, 400))
        z = np.ones_like(x)
        fes1 = -1 * (3 - x ** 2 - y) * (3 - 2 * (x ** 2) - y) <= 0
        fes2 = (3 - 0.625 * (x ** 2) - y) * (3 - 7 * (x ** 2) - y) <= 0
        fes3 = -1 * (1.62 - 0.18 * (x ** 2) - y) * (1.125 - 0.125 * (x ** 2) - y) <= 0
        fes4 = (2.07 - 0.23 * (x ** 2) - y) * (0.63 - 0.07 * (x ** 2) - y) <= 0
        z[fes1 & fes2 & fes3 & fes4 & (x ** 2 + y ** 2 >= 2)] = 0
        return x, y, z


class DisplayMW12(MW12):

    def __init__(self, n_var=15):
        super(DisplayMW12, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 2, 400), np.linspace(0, 2, 400))
        z = np.ones_like(x)
        fes1 = (1 - 0.8 * x - y + 0.08 * np.sin(2 * np.pi * (y - x / 1.5))) * \
               (1.8 - 1.125 * x - y + 0.08 * np.sin(2 * np.pi * (y / 1.8 - x / 1.6))) <= 0

        fes2 = -1 * (1 - 0.625 * x - y + 0.08 * np.sin(2 * np.pi * (y - x / 1.6))) * \
               (1.4 - 0.875 * x - y + 0.08 * np.sin(2 * np.pi * (y / 1.4 - x / 1.6))) <= 0

        fes3 = 0.8 * x + 0.08 * np.abs(np.sin(3.2 * np.pi * x)) + y >= 0.85
        z[fes1 & fes2 & fes3] = 0
        return x, y, z


class DisplayMW13(MW13):

    def __init__(self, n_var=15):
        super(DisplayMW13, self).__init__(n_var=n_var)

    @staticmethod
    def get_pf_region():
        x, y = np.meshgrid(np.linspace(0, 2, 400), np.linspace(0, 4.5, 400))
        z = np.ones_like(x)
        fes1 = (5 - np.exp(x) - 0.5 * np.sin(3 * np.pi * x) - y) * \
               (5 - (1 + 0.4 * x) - 0.5 * np.sin(3 * np.pi * x) - y) <= 0

        fes2 = -1 * (5 - (1 + x + 0.5 * (x ** 2)) - 0.5 * np.sin(3 * np.pi * x) - y) * \
               (5 - (1 + 0.7 * x) - 0.5 * np.sin(3 * np.pi * x) - y) <= 0

        fes3 = np.exp(x) + np.abs(0.5 * np.sin(3 * np.pi * x)) + y >= 5

        z[fes1 & fes2 & fes3] = 0
        return x, y, z


class DisplayMW14(MW14):

    def __init__(self, n_var=None, n_obj=3):
        super(DisplayMW14, self).__init__(n_var=n_var, n_obj=n_obj)
        self.nds = NonDominatedSorting()

    def get_pf_region(self):
        if self.n_obj == 2:
            x = np.linspace(0, 1.5, 100)[np.newaxis, :].T
            y = 6 - np.exp(x) - 1.5 * np.sin(1.1 * np.pi * (x ** 2))
            nd = self.nds.do(np.column_stack([x, y]), only_non_dominated_front=True)
            r = np.ones_like(x)
            r[nd] = 0
            x = x[r == 0]
            y = y[r == 0]
            z = np.column_stack([x, y])
            return x, y, z

        elif self.n_obj == 3:
            x, y = np.meshgrid(np.linspace(0, 1.5, 20), np.linspace(0, 1.5, 20))
            z = 0.5 * (12 - np.exp(x) - 1.5 * np.sin(1.1 * np.pi * (x ** 2)) -
                       np.exp(y) - 1.5 * np.sin(1.1 * np.pi * (y ** 2)))

            region = np.ones_like(z).reshape(z.size)
            temp_objs = np.column_stack([x.reshape(x.size)[:, np.newaxis], y.reshape(y.size)[:, np.newaxis],
                                         z.reshape(z.size)[:, np.newaxis]])
            nd = self.nds.do(temp_objs, only_non_dominated_front=True)
            region[nd] = 0
            region = region.reshape(z.shape)
            pf_index = (region == 0)
            z = z[pf_index]
            return x, y, z


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.ticker import LinearLocator

    A = DisplayMW1()
    X, Y, Z = A.get_pf_region()

    if A.n_obj == 3:
        X, Y, Z = A.get_pf_region()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # surf = ax.plot_surface(X, Y, Z, cmap=cm.gray,
        #                         linewidth=0, antialiased=False)
        surf = ax.plot_surface(X, Y, Z, color='gray', shade=False)
        plt.show()

    elif isinstance(A, DisplayMW14):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Z[:, 0], Z[:, 1])
        fig.show()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(Z, cmap=cm.gray,
                       origin='lower', extent=[0.5, 2.5, 0.5, 2.5],
                       vmax=abs(Z / 9).max(), vmin=-abs(Z).max())

        fig.show()





