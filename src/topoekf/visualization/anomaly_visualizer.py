import matplotlib.pyplot as plt
import numpy as np


class AnomalyVisualizer:
    def plot_scores(self, scores: np.ndarray):
        fig, ax = plt.subplots()
        ax.plot(scores)
        ax.axhline(-0.1, color="red", linestyle="--")
        ax.set_ylabel("anomaly score")
        return fig
