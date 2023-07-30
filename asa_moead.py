import numpy as np
from scipy.spatial.distance import cdist
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.algorithm import LoopwiseAlgorithm
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.core.variable import Real, get
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.reference_direction import default_ref_dirs
from surr_problem import SurrogateModel, SurrogateProblem, RBFModel, SurrProblemFDLS, SurrProblemCDLS

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
from pymoo.core.survival import split_by_feasibility
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.de import mut_binomial
from frofi import FROFI, FitnessSingle
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator


class NeighborhoodSelection(Selection):

    def __init__(self, prob=1.0) -> None:
        super().__init__()
        self.prob = Real(prob, bounds=(0.0, 1.0))

    def _do(self, problem, pop, n_select, n_parents, neighbors=None, **kwargs):
        assert n_select == len(neighbors)
        P = np.full((n_select, n_parents), -1)

        prob = get(self.prob, size=n_select)

        for k in range(n_select):
            if np.random.random() < prob[k]:
                P[k] = np.random.choice(neighbors[k], n_parents, replace=False)
            else:
                P[k] = np.random.permutation(len(pop))[:n_parents]

        return P


class ASAMOEAD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_neighbors=20,
                 decomposition=None,
                 prob_neighbor_mating=0.9,
                 sampling=LatinHypercubeSampling(),
                 crossover=SBX(prob=1.0, eta=20),
                 mutation=PM(prob_var=None, eta=20),
                 output=MultiObjectiveOutput(),
                 top_p_ddgs=10,
                 **kwargs):

        self.ref_dirs = ref_dirs

        # the decomposition metric used
        self.decomposition = decomposition

        # the number of neighbors considered during mating
        self.n_neighbors = n_neighbors

        self.neighbors = None

        self.selection = NeighborhoodSelection(prob=prob_neighbor_mating)

        # OSAM
        self.ref_pop =None

        self.top_p_ddgs = top_p_ddgs


        super().__init__(pop_size=len(ref_dirs),
                         sampling=sampling,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=NoDuplicateElimination(),
                         output=output,
                         advance_after_initialization=False,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        # if no reference directions have been provided get them and override the population size and other settings
        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        self.neighbors_outer = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, self.n_neighbors:]
        # if the decomposition is not set yet, set the default
        self.v1 = max(problem.n_var*200, 2000)

        self.surr_model = RBFModel(problem.n_var, problem.n_obj, problem.n_ieq_constr)


    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        ref_dirs = self.ref_dirs
        pop_F = self.pop.get('F')
        pop_index = calc_ref_assign(pop_F, ref_dirs)
        self.ref_pop = pop_index
        self.Archive = self.pop

    def _infill(self):

        for ref_id, ref_dir_i in enumerate(self.ref_dirs):
            unique_index = np.unique(self.Archive.get('X'), axis=0, return_index=True)[1]
            print(unique_index, len(unique_index), len(self.Archive))
            pop_data = self.Archive[unique_index]
            self.surr_model.fit(pop_data.get('X'), pop_data.get('F'), pop_data.get('G'))

            ref_i_pop = self.pop[self.ref_pop == ref_id]
            if len(ref_i_pop) == 0:
                opt_state = 's1'
            else:
                ref_i_pop_fea = ref_i_pop[ref_i_pop.get('feas')]
                if len(ref_i_pop_fea) == 0:
                    opt_state = 's2'
                else:
                    opt_state = 's3'

            if opt_state == 's1':
                cand = self.ddgs(ref_id, ref_dir_i, self.v1)
                print('ddgs', cand.get('X'))
                self.evaluator.eval(self.problem, cand)
                self.Archive = Population.merge(self.Archive, cand)
            elif opt_state == 's2':
                cand = self.fdls(ref_id)
                print(cand.get('X'))
                self.evaluator.eval(self.problem, cand)
                self.Archive = Population.merge(self.Archive, cand)
            else:
                cand = self.cdls(ref_id)
                print(cand.get('X'))
                self.evaluator.eval(self.problem, cand)
                self.Archive = Population.merge(self.Archive, cand)


            if self.judge_use_aiss(cand):
                cand_assis = self.AISS(ref_id, self.v1)
                self.evaluator.eval(self.problem, cand_assis)
                self.Archive = Population.merge(self.Archive, cand_assis)

            self.pop = RankAndCrowdingSurvival().do(self.problem, self.Archive, n_survive=self.pop_size)
            # judge terminal


    def fdls(self, ref_id):
        opt_pro = SurrProblemFDLS(self.problem.n_var, 1, self.problem.n_ieq_constr, self.surr_model,
                                  xl=self.problem.xl, xu=self.problem.xu, ref_inner=self.ref_dirs[self.neighbors[ref_id]],
                                  ref_outer=self.ref_dirs[self.neighbors_outer[ref_id]])

        fdls_init_pop = Population.new(X=self.pop.get('X'))
        Evaluator().eval(opt_pro, fdls_init_pop)
        opt_alg = FROFI(sampling=fdls_init_pop, pop_size=50)
        res = minimize(opt_pro, opt_alg, termination=('n_eval', 8000))
        print("fdls res X", res.X)
        res_pop_fitness = FitnessSingle(res.pop)
        sel_index = np.argmin(res_pop_fitness)
        cand_sel = res.pop[sel_index]
        cand_sel_x = cand_sel.get('X')
        # if len(res.X.shape) > 1:
        #     cand_sel_x = res.X[0]
        # else:
        #     cand_sel_x = res.X
        # print('fdls cand sel X:', cand_sel_x)
        return Population.new(X=cand_sel_x[None, :])

    def cdls(self, ref_id):
        opt_pro = SurrProblemCDLS(self.problem.n_var, 1, self.problem.n_ieq_constr, self.surr_model,
                                  xl=self.problem.xl, xu=self.problem.xu, ref=self.ref_dirs[ref_id],
                                  decomposition=self.decomposition)
        cdls_init_pop = Population.new(X=self.pop.get('X'))
        Evaluator().eval(opt_pro, cdls_init_pop)
        opt_alg = FROFI(sampling=cdls_init_pop, pop_size=50)
        res = minimize(opt_pro, opt_alg, termination=('n_eval', 8000))

        print("cdls res X", res.X)
        res_pop_fitness = FitnessSingle(res.pop)
        sel_index = np.argmin(res_pop_fitness)
        cand_sel = res.pop[sel_index]
        cand_sel_x = cand_sel.get('X')
        return Population.new(X=cand_sel_x[None, :])

    # use index choose neighborhood population
    def ddgs(self, ref_id, ref_use, v1):

        tao_i = Population.new()
        for ref_neighbor_i in self.neighbors[ref_id]:
            pop_neigh_i = self.pop[self.ref_pop == ref_neighbor_i]
            if len(pop_neigh_i) > 0:
                tao_i = Population.merge(tao_i, pop_neigh_i)

        ES = RankAndCrowdingSurvival().do(self.problem, tao_i, n_survive=self.top_p_ddgs)

        CS = Population.new()
        i_count = 1
        while len(CS) == 0:
            i_up = ref_id + i_count
            delta_i = self.pop[self.ref_pop == i_up]

            if len(delta_i) > 0:
                delta_i_fea = delta_i[delta_i.get('feas')]
                if len(delta_i_fea) > 0:
                    delta_i_fea_F = delta_i_fea.get('F')
                    delta_i_nd = delta_i_fea[NonDominatedSorting().do(delta_i_fea_F, only_non_dominated_front=True)]
                else:
                    delta_i_fea_cv = delta_i.get('cv')
                    delta_i_nd = delta_i[np.argsort(delta_i_fea_cv)[0]]

                CS = Population.merge(CS, delta_i_nd)

            i_down = ref_id - i_count
            delta_i = self.pop[self.ref_pop == i_down]
            if len(delta_i) > 0:
                delta_i_fea = delta_i[delta_i.get('feas')]
                if len(delta_i_fea) > 0:
                    delta_i_fea_F = delta_i_fea.get('F')
                    delta_i_nd = delta_i_fea[NonDominatedSorting().do(delta_i_fea_F, only_non_dominated_front=True)]
                else:
                    delta_i_fea_cv = delta_i.get('cv')
                    delta_i_nd = delta_i[np.argsort(delta_i_fea_cv)[0]]

                CS = Population.merge(CS, delta_i_nd)

            i_count += 1

        cand_pop_x = []
        for j in range(v1):
            x_cur = CS[np.random.randint(len(CS))]
            F1 = np.random.random() * 0.5 + 0.5
            F2 = np.random.random() * 0.5 + 0.5
            CR = np.random.random() * 0.5 + 0.5

            if len(ES) > 0:
                p_best = ES[np.random.randint(len(ES))]
                P_star = self.eliminate_duplicates.do(self.pop, x_cur)
                x_r = P_star[np.random.permutation(len(P_star))[:2]]
                v_j = x_cur.get('X') + F1 * (p_best.get('X') - x_cur.get('X')) + F2 * (x_r[0].get('X') - x_r[1].get('X'))
            else:
                P_star = self.eliminate_duplicates.do(self.pop, x_cur)
                x_r = P_star[np.random.permutation(len(P_star))[:3]]
                v_j = x_cur.get('X') + F1 * (x_r[0].get('X') - x_cur.get('X')) + F2 * (x_r[1].get('X') - x_r[2].get('X'))

            M = mut_binomial(1, self.problem.n_var, CR, at_least_once=True)[0]
            y_j = np.ones_like(x_cur.get('X'))
            try:
                y_j[M] = v_j[M]
                y_j[~M] = x_cur.get('X')[~M]
            except IndexError:
                print(M)
                print(y_j)
            cand_pop_x.append(y_j)


        cand_pop_x = np.vstack(cand_pop_x)
        pred_res = self.surr_model.evaluate(cand_pop_x)
        pred_F = pred_res['F']
        pred_G = pred_res['G']

        pop_cand = Population.new(X=cand_pop_x, F=pred_F, G=pred_G)

        cand_fea = pop_cand[pop_cand.get('feas')]
        if len(cand_fea) > 0:
            cand_fea_F = cand_fea.get('F')
            cand_fea_decmop_F = self.decomposition.do(cand_fea_F, ref_use)
            cand_sel = cand_fea[np.argmin(cand_fea_decmop_F)]

        else:
            cand_cv = pop_cand.get('cv')
            cand_sel = pop_cand[np.argmin(cand_cv)]
        print('ddgs cand X', cand_sel.get('X'))
        return Population.new(X=cand_sel.get('X')[None, :])


    def AISS(self, ref_id, v1):

        CS = Population.new()
        i_count = 1
        kernel_mat = kernel_matrix(self.pop.get('X'))
        pop_X = self.pop.get('X')
        while len(CS) == 0:
            i_up = ref_id + i_count
            delta_i = self.pop[self.ref_pop == i_up]

            if len(delta_i) > 0:
                delta_i_fea = delta_i[delta_i.get('feas')]
                if len(delta_i_fea) > 0:
                    delta_i_fea_F = delta_i_fea.get('F')
                    delta_i_nd = delta_i_fea[NonDominatedSorting().do(delta_i_fea_F, only_non_dominated_front=True)]
                else:
                    delta_i_fea_cv = delta_i.get('cv')
                    delta_i_nd = delta_i[np.argsort(delta_i_fea_cv)[0]]

                CS = Population.merge(CS, delta_i_nd)

            i_down = ref_id - i_count
            delta_i = self.pop[self.ref_pop == i_down]
            if len(delta_i) > 0:
                delta_i_fea = delta_i[delta_i.get('feas')]
                if len(delta_i_fea) > 0:
                    delta_i_fea_F = delta_i_fea.get('F')
                    delta_i_nd = delta_i_fea[NonDominatedSorting().do(delta_i_fea_F, only_non_dominated_front=True)]
                else:
                    delta_i_fea_cv = delta_i.get('cv')
                    delta_i_nd = delta_i[np.argsort(delta_i_fea_cv)[0]]

                CS = Population.merge(CS, delta_i_nd)

            i_count += 1

        cand_pop_x = []
        uncertain = []
        for j in range(v1):
            x_cur = CS[np.random.randint(len(CS))]
            F1 = np.random.random() * 0.5 + 0.5

            P_star = self.eliminate_duplicates.do(self.pop, x_cur)
            x_r = P_star[np.random.permutation(len(P_star))[:2]]
            v_j = x_cur.get('X') + F1 * (x_r[0].get('X') - x_r[1].get('X'))
            cand_pop_x.append(v_j)
            ker_vec_v_j = kernel_vector(v_j, pop_X)
            cand_uncertain = ker_vec_v_j[None, :].dot(kernel_mat).dot(ker_vec_v_j[:, None])[0, 0]
            uncertain.append(cand_uncertain)

        aiss_cand_x = cand_pop_x[np.argmax(uncertain)]
        return Population.new(X=aiss_cand_x[None, :])


    def judge_use_aiss(self, offspring):

        offspring_fea = offspring.get('feas')
        off_F = offspring.get('F')
        off_cv = offspring.get('cv')
        flag = False

        if offspring_fea:
            pop_fea, pop_infea = split_by_feasibility(self.pop, sort_infeas_by_cv=True)
            if np.sum(pop_fea) == 0:
                flag = True

            elif np.sum(pop_fea) == 1:
                pop_fea_F = self.pop[pop_fea].get('F')
                if np.all(off_F < pop_fea_F):
                    flag = True

            else:
                pop_fea_F = self.pop[pop_fea].get('F')
                if np.all(np.all(off_F < pop_fea_F, axis=1)):
                    flag = True

        else:
            pop_fea, pop_infea = split_by_feasibility(self.pop, sort_infeas_by_cv=True)
            if np.sum(pop_infea) > 0:
                pop_cv_max = self.pop[pop_infea][-1].get('cv')
                if off_cv < pop_cv_max:
                    flag = True

        return flag


    def _advance(self, infills=None, **kwargs):
        pass



def kernel_matrix(x):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    out = np.empty((x.shape[0], x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(i+1):
            out[i, j] = np.power(np.linalg.norm(x[i] - x[j]), 3)
            out[j, i] = out[i, j]

    return out


def kernel_vector(x_in, x):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    ker_vec = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        ker_vec[i] = np.power(np.linalg.norm(x_in - x[i]), 3)

    return ker_vec



def calc_ref_assign(F, ref_dirs):
    ref_dirs_norm = np.linalg.norm(ref_dirs, axis=1)
    d1s = F.dot(ref_dirs.T) / ref_dirs_norm
    F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
    d2s = np.sqrt(F_norm**2 - d1s**2)
    ref_index = np.argmin(d2s, axis=1)
    return ref_index

