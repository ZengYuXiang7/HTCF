from utils.utils import create_ipynb_file
order = []

for model in ['3']:
    for dim in [16, 32, 64, 128]:
        for density in [0.20]:
            string = f'!python Experiment.py --rounds {5} --per_or_all 1 --model {model} '
            string += f'--density {density} '
            string += f'--epochs {100} '
            string += f'--bs {256} --lr {0.001} --decay {0.001} '
            string += f'--dimension {dim} '
            string += f'--record {1} --program_test {0} --verbose {10} '
            string += f'--valid {0}'
            cell = {
                "cell_type" : "code",
                "source" : string,
                "metadata" : {}
            }
            order.append(cell)
create_ipynb_file(order, '超参数探索')
