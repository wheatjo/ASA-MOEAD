from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.problems.multi.mw import MW1, MW3
from pymoo.problems.multi.zdt import ZDT1
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from asa_moead import ASAMOEAD
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.algorithms.moo.moead import MOEAD
import pickle

pro = MW3(n_var=8)
ref_dir = UniformReferenceDirectionFactory(pro.n_obj, n_points=50).do()
alg = ASAMOEAD(ref_dirs=ref_dir, n_neighbors=10, decomposition=Tchebicheff(), sampling=LatinHypercubeSampling())
# alg = ASAMOEAD(ref_dirs=ref_dir)
res = minimize(problem=pro, algorithm=alg, termination=('n_eval', 960), verbose=True)

print(res)

with open("./mw3.pickle", 'wb') as f:
    pickle.dump({'res': res}, f)







