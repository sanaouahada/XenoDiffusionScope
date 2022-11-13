import argparse
import datetime
import os
import pickle
from multiprocessing import Pool
from typing import Union

import numpy as np
import xenodiffusionscope as xds
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=('Script to produce the LCE maps for all the hex centres.'))

parser.add_argument('-p', '--path',
                    help='Path to directory where to save patterns.',
                    type=str,
                    required=True)
parser.add_argument('-n', '--n_processes',
                    help=('Number of processed to use when making patterns.'
                          'Defaults to None, i.e. all the available cpus.'),
                    type=Union[None,int],
                    required=False,
                    default=None)

args = parser.parse_args()

def prepare():

    if not os.path.exists(args.path):
        print('Parsed path did not exist. Creating: ', args.path)
        os.makedirs(args.path, exist_ok=True)
    else:
        print('Found existing directory, (re-)writing at ', args.path)

    ### In the future to put in config file ###
    r_max, hex_size = 75, 1.56
    length = 2600
    liquid_gap = 5
    gas_gap = 5
    drift_field = 100
    x_bin_step = 1
    y_bin_step = 1
    n_traces = 1e8
    smooth_pattern = False
    ### ###

    Xenoscope = xds.TPC(r_max, length, liquid_gap, gas_gap, drift_field)
    mesh = Xenoscope.gate_mesh

    scintillation = xds.LCEPattern(Xenoscope)
    scintillation.define_pattern_props(x_bin_step = x_bin_step, 
                                       y_bin_step = y_bin_step, 
                                       n_traces = n_traces, 
                                       smooth_pattern = smooth_pattern,
                                       force_traces = True)

    initial_toy_pos = np.hstack(
        (mesh.hex_centers, 
        Xenoscope.liquid_level *np.ones((mesh.n_hexes,1))
        ))
    
    return scintillation, initial_toy_pos


def produce_pattern(input):
    hex_id,hex_position = input
    #print(f'Starting hex id: {hex_id}.')

    pattern = LCE_object.make_pattern_from_pos(hex_position[0],
                                                  hex_position[1],
                                                  hex_position[2])
    pattern_file_name = os.path.join(args.path, 'hex_%d.pck'%hex_id)
    with open(pattern_file_name, 'wb') as file:
        pickle.dump(pattern, file)
    #print(f'Saved pattern for hex id:  {hex_id}.')

if __name__ == '__main__':
    print('Starting pattern making at: ', datetime.datetime.now())
    LCE_object, initial_toy_pos = prepare()

    inputs = [(i , x) for (i ,x) in enumerate(initial_toy_pos)]
    #Create the mp object and run.
    pool = Pool(args.n_processes)

    for loop in tqdm(pool.imap(produce_pattern, inputs),
                     desc='Making LCE patterns: ',
                     total = len(inputs)):
        pass
    pool.close()
    pool.join()

    print('Finished pattern making at: ', datetime.datetime.now())