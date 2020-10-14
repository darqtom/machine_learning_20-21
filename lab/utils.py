import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker, cm


def get_fn_values(points, fn, X_vals):
    return np.array([fn(points, v) for v in X_vals])


def plot_1d_set(dataset, ax, loss_fns, show_title=False):
    linspace = np.linspace(dataset.min(), dataset.max(), num=200)
    ax.set_xlabel("v")
    ax.set_ylabel("Loss val")
    ax.scatter(dataset, [0] * len(dataset))
    for idx, loss_fn in enumerate(loss_fns):
        y_vals = get_fn_values(dataset, loss_fn, linspace)
        if show_title:
            ax.set_title(loss_fn.__name__)
        ax.plot(linspace, y_vals, label=loss_fn.__name__)

        
def plot_2d_set(dataset, ax, loss_fn):
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=40)
    second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=40)
    X, Y = np.meshgrid(first_linspace, second_linspace)
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[row_idx][col_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))
    ax.plot_surface(X, Y, Z)

    ax.scatter(dataset[:, 0], dataset[:, 1], np.zeros((dataset.shape[0],)))

    
def contour_2d_set(dataset, ax, loss_fn, linspaces=None):
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    if linspaces is None:
        first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=25)
        second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=25)
    else:
        first_linspace, second_linspace = linspaces
    X, Y = np.meshgrid(first_linspace, second_linspace, indexing="xy")
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[col_idx][row_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))
    
    ax.contour(X, Y, Z, levels=20)
    if linspaces is None:
        ax.scatter(dataset[:, 0], dataset[:, 1])
    else:
        ax.contourf(first_linspace, second_linspace, Z, levels=300, cmap=cm.PuBu_r)
    #    plt.colorbar()
        

def plot_2d_loss_fn(loss_fn, title, dataset):
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_2d_set(dataset, ax, loss_fn)
    ax = fig.add_subplot(1, 2, 2)
    contour_2d_set(dataset, ax, loss_fn)
    plt.show(fig)
    plt.close(fig)


def plot_minimums(dataset, loss_fns, loss_fns_mins, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    min_vals = []
    for (loss_fn, loss_fn_min, ax) in zip(loss_fns, loss_fns_mins, axes):
        min_val = loss_fn_min(dataset)
        min_vals += [min_val]
        ax.scatter(
            min_val,
            loss_fn(dataset, min_val),
            color="black"
        )
        plot_1d_set(dataset, ax, [loss_fn], show_title=True)

    plt.show(fig)
    plt.close(fig)
    print(
        "ME minimum: {:.2f} MSE minimum: {:.2f} Max Error minimum: {:.2f}".format(
            *min_vals)
    )


def plot_gradient_steps_1d(ax, dataset, gradient_descent_fn, grad_fn, loss_fn, num_steps=100, learning_rate=1e-1):
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate)
    plot_1d_set(dataset, ax, [loss_fn])
    y_vals = get_fn_values(dataset, loss_fn, all_v)
    ax.scatter(all_v, y_vals, c=np.arange(len(all_v)), cmap=plt.cm.viridis)
    return final_v


def plot_gradient_steps_2d(ax, dataset, gradient_descent_fn, grad_fn, loss_fn, num_steps=100, learning_rate=1e-2, linspaces=None):
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate)
    contour_2d_set(dataset, ax, loss_fn, linspaces)
    ax.scatter(all_v[:, 0], all_v[:, 1], c=np.arange(len(all_v)), cmap=plt.cm.viridis)

    print("Final grad value for {}: {}".format(loss_fn.__name__, final_grad))
    return final_v


def visualize_normal_dist(X, loc, scale):
    peak = 1 / np.sqrt(2 * np.pi * (scale ** 2))
    plt.hist(X, bins=50, density=True)
    plt.plot([loc - scale, loc - scale], [0, peak], color="r", label="1 sigma")
    plt.plot([loc + scale, loc + scale], [0, peak], color="r")

    plt.plot([loc - 2 * scale, loc - 2 * scale], [0, peak], color="b", label="2 sigma")
    plt.plot([loc + 2 * scale, loc + 2 * scale], [0, peak], color="b")

    plt.plot([loc - 3 * scale, loc - 3 * scale], [0, peak], color="g", label="3 sigma")
    plt.plot([loc + 3 * scale, loc + 3 * scale], [0, peak], color="g")
    plt.legend()

    
def scatter_with_whiten(X, whiten, name, standarize=False):
    plt.title(name)
    plt.scatter(X[:, 0], X[:, 1], label="Before whitening")
    white_X = whiten(X)
    plt.axis("equal")
    plt.scatter(white_X[:, 0], white_X[:, 1], label="After whitening")
    
    
    if standarize:
        X_standarized = (X - X.mean(axis=0)) / X.std(axis=0)
        plt.scatter(X_standarized[:, 0], X_standarized[:, 1], label="Standarized")
        
    plt.legend()
    plt.show()

