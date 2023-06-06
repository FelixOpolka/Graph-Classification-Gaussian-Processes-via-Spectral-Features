import pathlib
import pickle as pk
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    run_accs = []
    thresholds = np.linspace(0.01, 0.3, num=10)
    original_accs = []
    dataset = "mutag"
    model = "fourier"
    for run_idx in range(10):
        file_name = f"predictions_{model}/predictions_{dataset}_{run_idx}.pk"
        with pathlib.Path(file_name).open("rb") as fd:
            preds_val, preds_var_val, y_val, preds_test, preds_var_test, y_test = pk.load(fd)

        preds_var_val = preds_var_val if isinstance(preds_var_val, np.ndarray) else preds_var_val.numpy()
        preds_var_test = preds_var_test if isinstance(preds_var_test, np.ndarray) else preds_var_test.numpy()
        y_val = y_val if isinstance(y_val, np.ndarray) else y_val.numpy()
        y_test = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()

        if preds_var_val.shape[-1] > 1:
            preds_var_val = preds_var_val[np.arange(preds_val.shape[0]), preds_val]
            preds_var_test = preds_var_test[np.arange(preds_test.shape[0]), preds_test]
        else:
            preds_var_val = preds_var_val.reshape(-1)
            preds_var_test = preds_var_test.reshape(-1)
        # preds_var_val = np.mean(preds_var_val, axis=-1)
        # preds_var_test = np.mean(preds_var_test, axis=-1)

        acc = accuracy_score(y_test, preds_test)
        original_accs.append(acc)
        print(f"Original accuracy: {acc*100.0:.3f}")

        accs = []
        for threshold in thresholds:
            preds = preds_test[preds_var_test < threshold]
            labels = y_test[preds_var_test < threshold]
            if len(preds) > 0:
                accs.append(accuracy_score(labels, preds))
            else:
                accs.append(1.0)
            print(accs[-1])
        run_accs.append(accs)
    run_accs = np.stack(run_accs, axis=0)
    mean = np.mean(run_accs, axis=0)
    std = np.std(run_accs, axis=0)

    plt.figure(figsize=(6, 3.0))
    plt.xlabel("variance threshold")
    plt.ylabel("accuracy")
    plt.plot(thresholds, mean, linewidth=2.0)
    plt.fill_between(thresholds, mean - std, mean + std, alpha=0.4)
    plt.savefig(f"uncertainty_plot_{dataset}_{model}.png", bbox_inches="tight")
    plt.show()

    print("result for reference:", np.mean(original_accs))