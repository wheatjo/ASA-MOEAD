from displaymw import DisplayMW1
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
mpl.use('TkAgg')

with open("./mw1.pickle", 'rb') as f:
    data = pickle.load(f)


pro = DisplayMW1(n_var=8)

pop = data['res'].algorithm.Archive
pop_F = pop.get('F')

region_x, region_y, region = pro.get_pf_region()
fig, ax = plt.subplots(figsize=(10, 8))
pf = pro.pareto_front()
im = ax.imshow(region, cmap=cm.gray, origin='lower', extent=[region_x[0][0], region_x[0][-1], region_y[0][0], region_y[-1][0]],
           vmax=abs(region / 9).max(), vmin=-abs(region).max(), aspect='auto')
ax.plot(pf[:, 0], pf[:, 1], 'ko')
ax.plot(pop_F[:, 0], pop_F[:, 1], 'bo')

plt.show()
