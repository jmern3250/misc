import numpy as np 
import pickle 

with open('./Pickles/rcvrs.p', 'rb') as f:
    rcvrs_dict = pickle.load(f)
with open('./Pickles/sndrs.p', 'rb') as f:
    sndrs_dict = pickle.load(f)
with open('./Pickles/idx.p', 'rb') as f:
    idx = pickle.load(f)
with open('./Pickles/days.p', 'rb') as f:
    days = pickle.load(f)

with open('./Pickles/rcvrs2.p', 'wb') as f:
    pickle.dump(rcvrs_dict, f, protocol=2)
with open('./Pickles/sndrs2.p', 'wb') as f:
    pickle.dump(sndrs_dict, f, protocol=2)
with open('./Pickles/idx2.p', 'wb') as f:
    pickle.dump(idx, f, protocol=2)
with open('./Pickles/days2.p', 'wb') as f:
    pickle.dump(days, f, protocol=2)