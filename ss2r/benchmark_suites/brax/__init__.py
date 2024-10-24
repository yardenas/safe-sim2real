from ss2r.benchmark_suites.brax.cartpole import cartpole
from ss2r.benchmark_suites.rccar import rccar

randomization_fns = {
    "cartpole_swingup": cartpole.domain_randomization,
    "cartpole_swingup_sparse": cartpole.domain_randomization,
    "cartpole_balance": cartpole.domain_randomization,
    "inverted_pendulum": cartpole.domain_randomization,
    "rccar": rccar.domain_randomization,
}
