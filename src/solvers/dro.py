import numpy as np
from pyomo.environ import *


def dd_dro_model(W, b, R, T, alpha_0, x0, c, data, M, eps, theta):

    # Model
    m = ConcreteModel()

    # Index sets
    m.J_indxs = Set(initialize=range(W.shape[0]))
    m.tau_indxs = Set(initialize=range(T))
    m.p_indxs = Set(initialize=range(W.shape[1]))
    m.t_indxs = Set(initialize=[0])
    m.data_indxs = Set(initialize=range(data.shape[0]))


    # Variables
    ## Continuous
    m.p0 = Var(m.p_indxs, domain=NonNegativeReals, initialize=x0[:len(m.p_indxs)])
    m.s = Var(m.data_indxs, domain=NonNegativeReals, initialize=np.zeros(len(m.data_indxs))) #np.abs(np.random.randn(len(m.data_indxs)))
    m.t = Var(m.t_indxs, domain=Reals, initialize=[10])
    m.alpha = Var(m.p_indxs, domain=PositiveReals, initialize=alpha_0)
    #initialize=np.array([1 / len(m.p_indxs) for i in range(len(m.p_indxs))])) #sum to one?

    ## integer
    m.q = Var(m.data_indxs, within=NonNegativeReals, initialize=np.ones(len(m.data_indxs), dtype=bool))

    # Constraints
    ## Wasserstein ball
    def wasserstein_rule(m):
        N = len(m.data_indxs)
        return eps * N * m.t[0] - np.ones(N).dot(m.s) >= theta * N
    
    m.wasserstein_ball = Constraint(rule=wasserstein_rule)

    ## Sum participation factors
    def pf_rule(m):
        return np.dot(m.alpha, (np.ones(len(m.alpha)))) == 1

    m.pf_sum = Constraint(rule=pf_rule(m))

    ## Grid technical limits
    def dist_grid_rule(m, i, n, tau):
        
        return -(W[i,:].dot(m.alpha) * data[n, tau] - b[i] + W[i,:].dot(m.p0)) + M * m.q[n] * (np.abs(W[i,:].dot(m.alpha))) >= (m.t[0] - m.s[n]) * (np.abs(W[i,:].dot(m.alpha)))
        # return -(- b[i] + W[i,:].dot(m.p0)) / (np.abs(W[i,:].dot(m.alpha))) >= 0
    
    m.dist_grid = Constraint(m.J_indxs, m.data_indxs, m.tau_indxs, rule=dist_grid_rule)

    ## Grid ramp up/down
    def dist_rampup_rule(m, ng, n, tau):
        return -( m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n] * np.abs(m.alpha[ng]) >= (m.t[0] - m.s[n]) * np.abs(m.alpha[ng])
    def dist_rampdown_rule(m, ng, n, tau):
        return -(-m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n] * np.abs(m.alpha[ng]) >= (m.t[0] - m.s[n]) * np.abs(m.alpha[ng])

    m.rampup = Constraint(m.p_indxs, m.data_indxs, m.tau_indxs, rule=dist_rampup_rule)
    m.rampdown = Constraint(m.p_indxs, m.data_indxs, m.tau_indxs, rule=dist_rampdown_rule)

    ## Min prox
    def q_rule(m, n):
        return M * (1 - m.q[n]) >= m.t[0] - m.s[n]
    
    m.q_constraint = Constraint(m.data_indxs, rule=q_rule)

    ## Binary
    def q_bin_rule(m, n):
        return m.q[n] * (1 - m.q[n]) == 0.
    
    m.q_bin = Constraint(m.data_indxs, rule=q_bin_rule)

    # Objective function
    def objective_rule(m):
        return c[:len(m.p0)].dot(m.p0)
    
    m.obj = Objective(rule=objective_rule)

    # solver = SolverFactory('mindtpy')
    # solver = SolverFactory('octeract')
    
    solver = SolverFactory('ipopt')
    solver.options['tol'] = 1e-3
    # solver.options['max_iter'] = 10000
    return m, solver
