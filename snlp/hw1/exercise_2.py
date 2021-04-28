import numpy as np
from numpy.random import Generator, PCG64
import math
import matplotlib.pyplot as plt


G = Generator(PCG64())

def stick_breaking(n:int,alpha:float) -> np.ndarray:
  """
  Draws n samples from a stick-breaking process with beta distribution intensity alpha.

  :param n: number of samples
  :param alpha: intensity parameter of the beta distribution
  :returns: stick lengths
  """
  betas = G.beta(a=alpha, b=1.0, size=n)
  betas[1:] *= np.cumprod(1-betas[:-1])
  weights = np.sort(betas)[::-1]
  return weights

def plot_stick_lengths(stick_lengths:np.array,
		       alpha:float,
		       B:int) -> None:
  """
  Plots -log2(sticks)
  :param sticks: list of stick lenghts
  """
  fig, ax = plt.subplots()
  # word_pos = range(0, len(stick_lengths))
  stick_lengths_log = [-B*math.log2(x) for x in stick_lengths]
  ax.loglog(stick_lengths_log)
  # ax.loglog(word_pos, stick_lengths_log)
  ax.set_xlabel("rank")
  ax.set_ylabel("log(stick length)")
  ax.grid()
  plt.show()
  pass
