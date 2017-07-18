import tensorflow as tf
import numpy as np
import pdb

from model import *

if __name__ == '__main__':
    # pdb.set_trace()
    adc = ADC(layer_depths = [64,32],
        resolution = 14,
        learning_rate = 1e-3)

    adc.build_model()
    adc.train()