import nltk
import operator
import matplotlib.pyplot as plt

def analysis(name, data):
  """
  Plot Zipfian distribution of data + true Zipfian distribution. Compute MSE.

  :param name: title of the graph
  :param data: list of data
  """
  # print("TODO", name)
  
  frq_dict = nltk.FreqDist(data)
  
  # Sorting list in descending order of frequency
  frq_dict = dict(sorted(frq_dict.items(), key=operator.itemgetter(1), reverse=True))
  
  # Separate out word frequencies and word into lists.
  frq_list = list(frq_dict.values())
  word_list = list(frq_dict.keys())
  word_pos = range(0, len(word_list))
  # word_pos = range(1, len(word_list)+1) ## I think, this is correct, as log(0) is undefined.
  
  # import pdb; pdb.set_trace()
  
  fig, ax = plt.subplots()
  ax.loglog(word_pos, frq_list, label=str(name))
  ax.set_ylabel("log(frequency)")
  ax.set_xlabel("log(rank)")
  ax.set_title("Frequency Curve with log scaling: "+str(name))
  ax.grid()

  # Compute mean squared error
  # predicted_freq = [1/x for x in ]