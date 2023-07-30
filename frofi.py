from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems.multi.mw import MW9


class FROFI(GeneticAlgorithm):

    def __init__(self, sampling, pop_size):
        super(FROFI, self).__init__(sampling=sampling, pop_size=pop_size)


    def _infill(self):
        X = self.pop.get('X')
        pop_F = self.pop.get('F')[:, 0]
        best_ind_x = X[np.argmin(pop_F)]

        N, D = X.shape[0], X.shape[1]
        CR = np.array([0.1, 0.2, 1.0])
        cr_choice = np.random.choice(CR, N)
        CR = cr_choice[:, None]
        CR = np.tile(CR, (1, D))
        F = np.array([0.6, 0.8, 1.0])
        f_choice = np.random.choice(F, N)
        F = f_choice[:, None]
        F = np.tile(F, (1, D))

        P = np.argsort(np.random.random((N, N)), axis=1)
        P1 = X[P[:, 0]]
        P2 = X[P[:, 1]]
        P3 = X[P[:, 3]]
        PB = np.tile(best_ind_x, (N, 1))

        Rand = np.random.random((N, D))
        k1 = np.tile(np.random.random((N, 1)) < 0.5, (1, D))
        k2 = ~k1 & (np.random.random((N, D)) < CR)

        off_dec = np.copy(X)
        off_dec[k1] = X[k1] + Rand[k1] * (P1[k1] - X[k1]) + F[k1] * (P2[k1] - P3[k1])
        off_dec[k2] = P1[k2] + Rand[k2] * (PB[k2] - P1[k2]) + F[k2] * (P2[k2] - P3[k2])

        return Population.new(X=off_dec)

    def _advance(self, infills=None, **kwargs):

        pop_fitness = FitnessSingle(self.pop)
        off_fitness = FitnessSingle(infills)
        replace1 = pop_fitness > off_fitness
        replace2 = ~replace1 & (self.pop.get('F')[:, 0] > infills.get('F')[:, 0])
        Archive = infills[replace2]

        self.pop[replace1] = infills[replace1]

        # uu = np.round(max(5., self.problem.n_var / 2))
        # print(uu)
        Nf = np.round(len(self.pop) / np.round(max(5., self.problem.n_var / 2)))
        # get descend sort
        try:
            obj_temp = -1.0 * self.pop.get('F')[:, 0]
        except ValueError:
            print(self.pop)

        rank = np.argsort(obj_temp)
        self.pop = self.pop[rank]

        for i in range(int(np.floor(len(self.pop)/Nf))):
            if len(Archive) == 0:
                break
            else:
                current = np.arange(i*Nf+1, (i+1)*Nf, dtype=np.int32)
                worst = np.argmax(self.pop[current].get('cv'))
                best = np.argmin(Archive.get('cv'))
                if Archive[best].get('F')[0] < self.pop[current[worst]].get('F')[0]:
                    self.pop[current[worst]] = Archive[best]
                    Archive = np.delete(Archive, best)

        self.mutation(self.problem, self.pop)


    def mutation(self, problem, pop):
        pop_fea = pop.get('feas')
        if ~np.any(pop_fea):
            offDec = pop[np.random.randint(len(pop))].get('X')
            k = np.random.randint(problem.n_var)
            offDec[k] = np.random.uniform(problem.xl[k], problem.xu[k])
            off = Population.new(X=offDec[None, :])
            self.evaluator.eval(problem, off)
            worst = np.argmax(pop.get('cv'))
            if pop[worst].get('F')[0] > off.get('F')[0]:
                pop[worst] = off[0]


def FitnessSingle(pop):
    cv = pop.get('cv')
    f = pop.get('F')[:, 0]
    pop_fea = pop.get('feas')
    fitness = np.ones_like(cv)
    fitness[pop_fea] = f[pop_fea]
    fitness[~pop_fea] = cv[~pop_fea] + 1e10
    return fitness



if __name__ == '__main__':

    from pymoo.problems.single.g import G3
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    pro = G3()
    pop_init = FloatRandomSampling().do(problem=pro, n_samples=100)
    alg = FROFI(pop_size=100, sampling=pop_init)
    res = minimize(problem=pro, algorithm=alg, termination=('n_gen', 1000), verbose=True)
    print(res.X, res.F)




