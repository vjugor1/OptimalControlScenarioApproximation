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
    s_init = 1*np.ones(len(m.data_indxs))
    m.s = Var(m.data_indxs, domain=NonNegativeReals, initialize=s_init) #np.abs(np.random.randn(len(m.data_indxs)))
    m.t = Var(m.t_indxs, domain=Reals, initialize=[sum(s_init)/(eps * len(m.data_indxs)) + theta / eps])
    m.alpha = Var(m.p_indxs, domain=PositiveReals, initialize=alpha_0)
    #initialize=np.array([1 / len(m.p_indxs) for i in range(len(m.p_indxs))])) #sum to one?

    ## integer
    # m.q = Var(m.data_indxs, within=NonNegativeReals, initialize=np.zeros(len(m.data_indxs)))
    m.q = Var(m.data_indxs, within=Binary, initialize=np.ones(len(m.data_indxs)))

    # Constraints
    ## Wasserstein ball
    def wasserstein_rule(m):
        N = len(m.data_indxs)
        return eps * N * m.t[0] - np.ones(N).dot(m.s) >= theta * N
    
    m.wasserstein_ball = Constraint(rule=wasserstein_rule)

    ## Sum participation factors
    def pf_rule(m):
        return np.dot(m.alpha, (np.ones(len(m.alpha)))) == 1
    
    def pf_rule_vicinity(m, i):
        return np.abs(m.alpha[i] - 1 / len(m.alpha)) <= 0.2

    # m.pf_vicinity = Constraint(m.p_indxs, rule=pf_rule_vicinity)

    m.pf_sum = Constraint(rule=pf_rule)

    ## Grid technical limits
    def dist_grid_rule(m, i, n, tau):
        
        # return -(W[i,:].dot(m.alpha) * data[n, tau] - b[i] + W[i,:].dot(m.p0)) + M * m.q[n] * (np.abs(W[i,:].dot(m.alpha))) >= (m.t[0] - m.s[n]) * (np.abs(W[i,:].dot(m.alpha)))
        return -(W[i,:].dot(m.alpha) * data[n, tau] - b[i] + W[i,:].dot(m.p0)) + M * m.q[n]  >= (m.t[0] - m.s[n]) 
        # return -(W[i,:].dot(m.alpha) * data[n, tau] - b[i] + W[i,:].dot(m.p0)) / (np.abs(W[i,:].dot(m.alpha))) + M * m.q[n]  >= (m.t[0] - m.s[n]) 
    
    m.dist_grid = Constraint(m.J_indxs, m.data_indxs, m.tau_indxs, rule=dist_grid_rule)

    ## Grid ramp up/down
    def dist_rampup_rule(m, ng, n, tau):
        # return -( m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n] * np.abs(m.alpha[ng]) >= (m.t[0] - m.s[n]) * np.abs(m.alpha[ng])
        # return -( m.alpha[ng] * data[n, tau] - R[ng]) / np.abs(m.alpha[ng]) + M * m.q[n] >= (m.t[0] - m.s[n])
        return -( m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n]  >= (m.t[0] - m.s[n]) 
        # return -( m.alpha[ng] * data[n, tau] - R[ng]) / np.abs(m.alpha[ng])  >= 0
    def dist_rampdown_rule(m, ng, n, tau):
        # return -(-m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n] * np.abs(m.alpha[ng]) >= (m.t[0] - m.s[n]) * np.abs(m.alpha[ng])
        return -(-m.alpha[ng] * data[n, tau] - R[ng]) + M * m.q[n]  >= (m.t[0] - m.s[n]) 
        # return -(-m.alpha[ng] * data[n, tau] - R[ng]) / np.abs(m.alpha[ng])  >= 0

    m.rampup = Constraint(m.p_indxs, m.data_indxs, m.tau_indxs, rule=dist_rampup_rule)
    m.rampdown = Constraint(m.p_indxs, m.data_indxs, m.tau_indxs, rule=dist_rampdown_rule)

    ## Min prox
    def q_rule(m, n):
        return M * (1 - m.q[n]) >= m.t[0] - m.s[n]
    
    m.q_constraint = Constraint(m.data_indxs, rule=q_rule)

    ## Binary
    def q_bin_rule(m, n):
        return m.q[n] * (1 - m.q[n]) == 0.
    def q_bin_rule_1(m, n):
        return m.q[n] <= 1.
    def q_bin_rule_2(m, n):
        return m.q[n] >= 0.
    def q_bin_rule_3(m, n):
        return m.q[n] >= 0.99
    def q_bin_rule_4(m, n):
        return m.q[n] <= 0.01
    # m.q_bin1 = Constraint(m.data_indxs, rule=q_bin_rule_1)
    # m.q_bin2 = Constraint(m.data_indxs, rule=q_bin_rule_2)
    # m.q_bin3 = Constraint(m.data_indxs, rule=q_bin_rule_3)
    # m.q_bin4 = Constraint(m.data_indxs, rule=q_bin_rule_4)
    # m.q_bin = Constraint(m.data_indxs, rule=q_bin_rule)

    # Objective function
    def objective_rule(m):
        return c[:len(m.p0)].dot(m.p0)
    
    m.obj = Objective(rule=objective_rule)

    solver = SolverFactory('mindtpy')
    # solver = SolverFactory('octeract')
    
    # solver = SolverFactory('ipopt')
    # solver.options['tol'] = 1e-3
    # solver.options['max_iter'] = 10000
    return m, solver
