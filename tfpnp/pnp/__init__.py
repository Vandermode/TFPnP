# from .solver_csmri import ADMMSolver_CSMRI, HQSSolver_CSMRI, PGSolver_CSMRI, APGSolver_CSMRI, REDADMMSolver_CSMRI
# from .solver_ct import ADMMSolver_CT, PGSolver_CT
# from .solver_pr import ADMMSolver_PR, PGSolver_PR


# def get_solver(opt):
#     print('[i] use solver: {}'.format(opt.solver))
#     if opt.solver == 'admm':
#         solver = ADMMSolver_CSMRI(opt)
#     elif opt.solver == 'hqs':
#         solver = HQSSolver_CSMRI(opt)
#     elif opt.solver == 'pg':
#         solver = PGSolver_CSMRI(opt)
#     elif opt.solver == 'apg':
#         solver = APGSolver_CSMRI(opt)
#     elif opt.solver == 'pd':
#         # solver = PDSolver_CSMRI(opt)
#         pass
#     elif opt.solver == 'pdhg':
#         # solver = PDHGSolver_CSMRI(opt)
#         pass
#     elif opt.solver == 'redadmm':
#         solver = REDADMMSolver_CSMRI(opt)
#     else:
#         raise NotImplementedError

#     return solver


# def get_solver_ct(opt):
#     print('[i] use solver: {}'.format(opt.solver))
#     if opt.solver == 'admm':
#         solver = ADMMSolver_CT(opt)
#     elif opt.solver == 'pg':
#         solver = PGSolver_CT(opt)
#     else:
#         raise NotImplementedError

#     return solver


# def get_solver_pr(opt):
#     print('[i] use solver: {}'.format(opt.solver))
#     if opt.solver == 'admm':
#         solver = ADMMSolver_PR(opt)
#     elif opt.solver == 'pg':
#         solver = PGSolver_PR(opt)
#     else:
#         raise NotImplementedError

#     return solver
