from read_data import read_data
import pickle

docs, target_summaries, stories = read_data('combined copy.txt')
with open ('amr_proxy_dict.txt', 'wb') as out:
    pickle.dump(docs, out)