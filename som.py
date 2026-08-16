# som.py — reconstructed from som.pyc (Python 3.12)
# Original author: Francisco Serradilla, Universidad Politécnica de Madrid

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import collections


class SOM:
    """
    Class SOM:
        N, M: size of the competitive layer
        metric: SOM.euclidean (default), SOM.manhattan, SOM.nmin, SOM.nmax
        toroidal: use toroidal structure (default: False)
        square: use square neighborhood
    """

    # ── Static metrics ────────────────────────────────────────────────────────
    @staticmethod
    def euclidean(v):
        return np.linalg.norm(v)

    @staticmethod
    def manhattan(v):
        return np.linalg.norm(v, 1)

    @staticmethod
    def nmin(v):
        return np.linalg.norm(v, -np.inf)

    @staticmethod
    def nmax(v):
        return np.linalg.norm(v, np.inf)

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self, N, M, metric=None, toroidal=False, square=False):
        self.N        = N
        self.M        = M
        self.metric   = metric if metric is not None else SOM.euclidean
        self.toroidal = toroidal
        self.square   = square
        self._center  = min(N, M) // 2
        self.ninputs  = 0

        if self.square:
            self.neighborhood = self.square_map
        else:
            self.neighborhood = self.sigma_map

    # ── Neighborhood functions ────────────────────────────────────────────────
    def square2d(self, x, y, mux, muy, sig):
        return np.where(np.abs(x - mux) <= sig, 1, 0) * \
               np.where(np.abs(y - muy) <= sig, 1, 0)

    def gaussian2d(self, x, y, mux, muy, sig):
        return np.exp(-(np.power(x - mux, 2) + np.power(y - muy, 2)) / (2.0 * sig ** 2))

    def square_map(self, indexw, sigma):
        x, y = np.meshgrid(np.arange(self.M), np.arange(self.N))
        return self.square2d(x, y, indexw[0], indexw[1], sigma).astype(float)

    def sigma_map(self, indexw, sigma):
        y, x = np.meshgrid(np.arange(self.M), np.arange(self.N))
        return self.gaussian2d(x, y, indexw[0], indexw[1], sigma)

    def toroidal_map(self, indexw, sigma, func):
        center = np.array(self._center)
        return np.roll(func(indexw, sigma), center)

    # ── Winner ────────────────────────────────────────────────────────────────
    def winner(self, sample):
        reshaped    = self.W.reshape(self.ninputs, self.N * self.M)
        input_tiled = np.vstack([sample] * (self.N * self.M)).T
        diff        = input_tiled - reshaped
        activations = np.apply_along_axis(self.metric, 0, diff)
        index       = np.argmin(activations)
        qerror      = activations[index]
        realindex   = (index // self.M, index % self.M)
        return (realindex, diff, qerror)

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, w, alpha, sigma):
        reshaped = w[1].reshape(self.ninputs, self.N, self.M)
        if self.toroidal:
            sigmas = self.toroidal_map(w[0], sigma, self.neighborhood)
        else:
            sigmas = self.neighborhood(w[0], sigma)
        delta   = alpha * reshaped * \
                  np.vstack([sigmas] * self.ninputs).reshape(self.ninputs, self.N, self.M)
        self.W += delta

    # ── Decay functions ───────────────────────────────────────────────────────
    def setdecay(self, sigma0, sigmaf, epochs, k=1.3):
        self.k   = k
        self.tau = epochs / (np.log(sigma0 / sigmaf) ** (1 / k))

    def edecay(self, sigma0, t):
        return sigma0 * np.exp(-(t / self.tau) ** self.k)

    def ldecay(self, pmax, pmin, t, maxite):
        return (maxite - t) * (pmax - pmin) / maxite + pmin

    # ── Cluster distance (quantization error) ────────────────────────────────
    def cluster_distance(self):
        reshaped = self.W.reshape(self.ninputs, self.N * self.M)
        cm       = np.mean(reshaped, axis=1)
        d        = np.linalg.norm(reshaped - cm.reshape(-1, 1), axis=0)
        return np.mean(d)

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, samples, epochs=100, alpha0=0.8, alphaf=0.05,
              sigmaf=0.1, auto_stop=0.0001, trace=100, verbose=1):
        """
        Train a SOM, initializing weights and neighborhood.
        samples : vectors to train with
        epochs  : epoch number
        alpha0  : initial learning rate
        alphaf  : final learning rate value
        sigmaf  : final neighborhood value
        auto_stop: stop if quantization error below this (default: 0.0001)
        trace   : epoch to trace (default: 100)
        verbose : trace level (default 1)
        """
        if self.N < 1 or self.M < 1:
            raise Exception('Invalid SOM size')

        self.ninputs  = len(samples[0])
        self.nsamples = len(samples)

        # Initialize weights: uniform in [-0.1, 0.1]
        self.W = np.random.rand(self.ninputs, self.N, self.M) * 0.2 - 0.1

        sigma0 = max(self.N, self.M)
        self.setdecay(sigma0, sigmaf, epochs, k=1.3)
        self.t0 = time.time()

        def print_trace():
            print(f"ite: {e:3d}, mean_qe: {qerror / self.nsamples:.3f}"
                  f"; max_qe: {max_qerror:.3f}"
                  f"; d: {self.cluster_distance():.3f}"
                  f"; alpha: {alpha:.2f}"
                  f"; sigma: {sigma:.2f}"
                  f"; lapse: {time.time() - self.t0:.2f}")
            self.t0 = time.time()
            if verbose > 1:
                print(self.activation_map(samples))

        oldqerror = math.inf
        for e in range(1, epochs + 1):
            qerror    = 0.0
            max_qerror = 0.0
            alpha     = self.ldecay(alpha0, alphaf, e, epochs)
            sigma     = self.edecay(sigma0, e)

            for i in range(self.nsamples):
                w = self.winner(samples[i])
                qerror    += w[2]
                if w[2] > max_qerror:
                    max_qerror = w[2]
                self.update(w, alpha, sigma)

            if trace and e % trace == 0:
                print_trace()

            if auto_stop != 0.0 and qerror / self.nsamples < auto_stop:
                break
            else:
                oldqerror = qerror

        print_trace()

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, samples):
        """
        Get the class for new data.
        samples: the new data
        Returns list of (row, col) tuples.
        """
        res = []
        for sample in samples:
            w = self.winner(sample)
            res.append(w[0])
        return res

    # ── Prototypes ────────────────────────────────────────────────────────────
    def prototypes(self):
        """Return all the class prototypes."""
        return self.W.reshape(self.ninputs, self.N * self.M).T

    # ── Activation map ────────────────────────────────────────────────────────
    def activation_map(self, samples):
        """
        Return the number of samples in each cluster.
        """
        p   = self.predict(samples)
        res = np.zeros((self.N, self.M), dtype='int')
        for w in p:
            res[w[0]][w[1]] += 1
        return res

    # ── Distance map ─────────────────────────────────────────────────────────
    def distance_map(self):
        dR = np.zeros((self.N, self.M))
        dB = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M - 1):
                dR[i, j] = self.metric(self.W[:, i, j] - self.W[:, i, j + 1])
        for i in range(self.N - 1):
            for j in range(self.M):
                dB[i, j] = self.metric(self.W[:, i, j] - self.W[:, i + 1, j])
        return dR, dB

    # ── Distribution ─────────────────────────────────────────────────────────
    def distribution(self, samples, labels=None):
        """
        For each cluster, return list of the labels in that cluster.
        """
        self.ninputs = len(samples[0])
        if labels is None:
            labels = range(len(samples))
        elif type(labels) is np.ndarray:
            labels = labels.tolist()

        p   = self.predict(samples)
        map_ = [[[] for _ in range(self.M)] for _ in range(self.N)]
        for i in range(len(p)):
            item = map_[p[i][0]][p[i][1]]
            item.append(labels[i])

        return [[sorted(map_[i][j]) for j in range(self.M)] for i in range(self.N)]

    # ── Summarize distribution ────────────────────────────────────────────────
    def summarize_distribution(self, X, labels):
        """
        Create a set of labels summarizing the assignments of labels to classes.
        """
        from collections import Counter

        def summary(numbers):
            if not numbers:
                return 'N/A: 0.00%'
            counts        = Counter(numbers)
            most_frequent, frequency = counts.most_common(1)[0]
            probability   = frequency / len(numbers) * 100
            return f"{most_frequent}\n{probability:.1f}%"

        items = self.distribution(X, labels)
        l     = [[summary(inner_element) for inner_element in inner_list]
                 for inner_list in items]
        return np.array(l)

    # ── Labels ────────────────────────────────────────────────────────────────
    def labels(self):
        """Create labels for each neuron based on centroids."""
        res = []
        for i in range(self.N):
            row = []
            for j in range(self.M):
                parts = ['[ ']
                for k in range(self.ninputs):
                    parts.append(f'{self.W[k, i, j]:.2f} ')
                parts.append(']')
                row.append(''.join(parts))
            res.append(row)
        return res

    # ── Draw map ──────────────────────────────────────────────────────────────
    def draw_map(self, samples, labels='auto', size=(800, 600),
                 textual=True, colors='summer'):
        """
        Create a graphical representation of the map with labels,
        number of elements in cluster, and frontiers (distances).
        samples : data to draw
        labels  : label for each neuron (use 'auto' to compute from data,
                  or pass a numpy array of shape (N, M))
        size    : figure size in pixels (default: (800, 600))
        textual : draw textual info (default: True)
        colors  : colormap (default: 'summer')
        """
        plt.ion()
        px   = 1 / plt.rcParams['figure.dpi']
        fig  = plt.figure(figsize=(size[0] * px, size[1] * px))
        axes = fig.add_subplot(111)
        plt.gca().invert_yaxis()

        amap   = np.array(self.activation_map(samples))

        # Determine levels for the heatmap
        if type(labels) is np.ndarray and type(labels[0, 0]) is not np.str_:
            levels = labels
        else:
            levels = amap

        heatmap = axes.pcolormesh(levels, cmap=colors, alpha=1.0,
                                  vmax=levels.max(), vmin=levels.min())
        plt.colorbar(heatmap)

        # Compute text labels
        if type(labels) is str and labels == 'auto':
            labels = self.summarize_distribution(samples, None)

        textbox = dict(boxstyle='round', facecolor='w',
                       edgecolor='none', alpha=0.5)

        for i in range(self.N):
            for j in range(self.M):
                if textual:
                    with np.printoptions(precision=3, suppress=True):
                        print(self.W[:, i, j], end=' ')
                margin = 0
                if type(labels) is np.ndarray:
                    plt.text(j + 0.5, i + 0.65,
                             '%s' % labels[i][j],
                             ha='center', va='center',
                             bbox=textbox, c='k')
                    margin = 0.3
                plt.text(j + 0.5, i + 0.5 - margin,
                         '%.0f' % amap[i, j],
                         ha='center', va='center',
                         bbox=textbox, c='k')
            if textual:
                print()

        # Draw frontiers
        dR, dB = self.distance_map()
        scale  = 5 / max(dR.max(initial=0), dB.max(initial=0))
        color  = '#ff8080'

        for i in range(self.N):
            for j in range(self.M - 1):
                plt.plot([j + 1, j + 1], [i, i + 1],
                         color, linewidth=scale * dR[i, j])

        for i in range(self.N - 1):
            for j in range(self.M):
                plt.plot([j, j + 1], [i + 1, i + 1],
                         color, linewidth=scale * dB[i, j])

    # ── Draw mesh ─────────────────────────────────────────────────────────────
    def draw_mesh(self, samples, size=(800, 600),
                  draw_samples=False, draw_number=False):
        """
        Draw prototype grid. Only valid for 2D inputs.
        """
        if self.ninputs > 2:
            raise Exception('Not available to input dimension > 2')
        plt.ion()
        px   = 1 / plt.rcParams['figure.dpi']
        fig  = plt.figure(figsize=(size[0] * px, size[1] * px))
        axes = fig.add_subplot(111)
        p    = self.prototypes()
        xlim = axes.get_xlim()

        for i in range(self.N):
            for j in range(self.M - 1):
                axes.plot([self.W[0, i, j], self.W[0, i, j + 1]],
                          [self.W[1, i, j], self.W[1, i, j + 1]], 'b+')
        for i in range(self.N - 1):
            for j in range(self.M):
                axes.plot([self.W[0, i, j], self.W[0, i + 1, j]],
                          [self.W[1, i, j], self.W[1, i + 1, j]], 'b+')

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, name):
        """Save SOM. name: file name"""
        with open(name, 'wb') as f:
            np.save(f, self.W)
            np.save(f, self.M)
            np.save(f, self.N)
            np.save(f, self.ninputs)
            np.save(f, self.nsamples)

    def load(self, name):
        """Load SOM. name: file name"""
        with open(name, 'rb') as f:
            self.W        = np.load(f)
            self.M        = int(np.load(f))
            self.N        = int(np.load(f))
            self.ninputs  = np.load(f)
            self.nsamples = np.load(f)

    # ── Describe ──────────────────────────────────────────────────────────────
    def describe(self):
        """Describe a SOM."""
        print(f'Competitive map of {self.N} x {self.M}')
        print(f'Metric is {self.metric}')
        if self.toroidal:
            print('toroidal is True')
        if self.square:
            print('square is True')
        if self.ninputs > 0:
            print(f'som has {self.ninputs} inputs')
            print(f'som was trained with {self.nsamples} samples')
        else:
            print('som untrained')

    # ── Report ────────────────────────────────────────────────────────────────
    def report(self, samples, labels):
        d = self.distribution(samples, labels)
        W = np.array([[self.metric(self.W[:, i, j])
                       for j in range(self.M)]
                      for i in range(self.N)])
        result = []
        for i in range(self.N):
            for j in range(self.M):
                best  = np.inf
                blabel = -1
                for item in d[i][j]:
                    dist = self.metric(self.W[:, i, j])
                    if dist < best:
                        best   = dist
                        blabel = int(item)
                result.append(blabel)
        return result
