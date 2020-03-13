import json
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
path = "data/execution_engine.pt.json"

results = json.load(open(path, "r"))


def smooth(x, y):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(x.min(), x.max(), 300)

    spl = make_interp_spline(x, y, k=3)
    return xnew, spl(xnew)

x, y = smooth(np.array(results['train_losses_ts']), results['train_losses'])
plt.plot(x, y, label="train losses")

plt.legend()
plt.xlabel("Training Iterations")
plt.ylabel("Training Loss")
plt.savefig("Losses")

plt.clf()
plt.plot(results['val_accs'], label="val accs")
plt.plot(results['train_accs'], label="train accs")
plt.legend()

plt.xlabel("Training Iterations")
plt.ylabel("Accuracy")
plt.savefig("accuracy")
