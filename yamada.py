import topoly

result = topoly.yamada(
    chain="theta_curve.txt.gz",
    closure=3,
    tries=1000,
    reduce_method=1,
    cuda=False,
    run_parallel=False,
    debug=True
)

print(result)