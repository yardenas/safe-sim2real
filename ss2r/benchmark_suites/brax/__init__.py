from ss2r.benchmark_suites.brax.cartpole import cartpole

randomization_fns = {
    "cartpole_swingup": cartpole.domain_randomization_length,
    "cartpole_swingup_sparse": cartpole.domain_randomization,
    "cartpole_balance": cartpole.domain_randomization,
    "inverted_pendulum": cartpole.domain_randomization,
}
