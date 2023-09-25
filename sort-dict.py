# Creates a sorted dictionary (sorted by key)
from collections import OrderedDict
import numpy as np

dict = {'kmer1': 108.0, 'kmer2': 42.0, 'kmer3': 40.0, 'kmer4': 42.0, 'kmer5': 37.0, 'kmer6': 29.0, 'kmer7': 38.0, 'kmer8': 45.0, 'kmer9': 42.0, 'kmer10': 33.0, 'kmer11': 37.0, 'kmer12': 43.0, 'kmer13': 34.0, 'kmer14': 27.0, 'kmer15': 28.0, 'kmer16': 35.0, 'kmer17': 82.0, 'kmer18': 34.0, 'kmer19': 22.0, 'kmer20': 56.0, 'kmer21': 40.0, 'kmer22': 55.0, 'kmer23': 33.0, 'kmer24': 31.0, 'kmer25': 71.0, 'kmer26': 57.0, 'kmer27': 43.0, 'kmer28': 35.0, 'kmer29': 41.0, 'kmer30': 36.0, 'kmer31': 29.0, 'kmer32': 27.0, 'kmer33': 58.0, 'kmer34': 40.0, 'kmer35': 51.0, 'kmer36': 45.0, 'kmer37': 71.0, 'kmer38': 61.0, 'kmer39': 22.0, 'kmer40': 48.0, 'kmer41': 55.0, 'kmer42': 45.0, 'kmer43': 92.0, 'kmer44': 62.0, 'kmer45': 74.0, 'kmer46': 34.0, 'kmer47': 23.0, 'kmer48': 53.0, 'kmer49': 53.0, 'kmer50': 78.0, 'kmer51': 36.0, 'kmer52': 54.0, 'kmer53': 28.0, 'kmer54': 22.0, 'kmer55': 25.0, 'kmer56': 50.0, 'kmer57': 52.0, 'kmer58': 47.0, 'kmer59': 24.0, 'kmer60': 36.0, 'kmer61': 43.0, 'kmer62': 29.0, 'kmer63': 58.0, 'kmer64': 54.0}
print(dict)

keys = list(dict.keys())
values = list(dict.values())
sorted_value_index = np.argsort(values)
sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

print(sorted_dict)