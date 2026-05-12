import matplotlib.pyplot as plt
import numpy as np


class TDAVisualizer:
    def plot_persistence_diagram(self, dgm: np.ndarray):
        fig, ax = plt.subplots()
        if len(dgm):
            ax.scatter(dgm[:, 0], dgm[:, 1])
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        return fig
