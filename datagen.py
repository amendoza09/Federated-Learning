import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(w, b, datasets):
    # plot the dataset
    fig, ax = plt.subplots()

    global_ds = np.vstack(datasets)
    for i, dataset in enumerate(datasets):
        x = dataset[:, 0]
        y = dataset[:, 1]

        ax.plot(x, y, '.', markersize=12, label=f'client {i}')

    # generate predictions using w and b
    x = global_ds[:, 0]
    predictions = np.polyval([w, b], x)

    ax.plot(x, predictions, '-', marker=None)
    regf_str = "$y={0:.5f}x + {1:.5f}$".format(w, b)
    fig.suptitle(regf_str, fontsize=20)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.legend()

    return fig


def main(args):
    data_sizes = [50, 30, 80]
    sigma = [0.8, 0.2, 1.2]
    n_clients = len(data_sizes)
    n_points = sum(data_sizes)
    w, b = args.slope, args.bias
    x = np.linspace(args.min_val, args.max_val, n_points)
    y = np.polyval([w, b], x)
    noise = np.concatenate([sigma[i] * np.random.randn(data_sizes[i])
                            for i in range(n_clients)])
    noisy_y = y + noise

    df = pd.DataFrame({'x': x, 'y': noisy_y})
    np_datasets = []
    start_idx = 0
    for i in range(n_clients):
        end_idx = start_idx + data_sizes[i]
        dset = df[start_idx:end_idx]
        dset.to_csv(f'dataset_client{i}.csv', index=False)
        np_datasets.append(dset.to_numpy())
        start_idx = end_idx

    fig = visualize_data(w, b, np_datasets)
    fig.savefig('dataset.png')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_val', type=float, default=-5, help='minimum x-value')
    parser.add_argument('--max_val', type=float, default=5, help='maximum x-value')
    parser.add_argument('--slope', type=float, default=2, help='slope of regression function')
    parser.add_argument('--bias', type=float, default=1, help='bias or intercept')
    parser.add_argument('--sigma', type=float, default=0.5, help='std of noise')
    args = parser.parse_args()

    main(args)
