from playground.maml.maml_torch.archive.maml_torch import G

# experiment 1.
params = []
for G.seed in range(1000, 1020):
    for G.k_shot in [5, 10, 20]:
        for G.n_gradient_steps in [1, 5, 10, 20]:
            G.log_prefix = 'maml_baselines/maml_torch/sinusoid-k_shot_{k_shot}-n_grad_{n_gradient_steps}-seed_{seed}'.format(
                **vars(G))
            params.append(vars(G))

print(len(params))
