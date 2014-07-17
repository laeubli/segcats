#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob, argparse
from adaptors.xml import *
from fileIO import serialiseObservationSequence

def getArgumentParser():
    """
    Return an ArgumentParser object to handle the arguments provided
    to this script.
    """
    parser = argparse.ArgumentParser(description="Extracts observations of multiple event types (e.g., keystrokes, mouse clicks, and eye fixations) from 1..* XML files. The observations are suitable to train a GMM HMM. The types of features to be extracted can be specified by choosing a suitable adaptor (run this program with --help for more information).")
    parser.add_argument("source", help="The XML file(s) from which to extract observations. Multiple files can be passed via wildcards, e.g., P*.xml.", type=str, nargs='*')
    parser.add_argument("-o", "--output_dir", help="The target folder where the observation sequences (one CSV file per input XML file) will be stored. The default is the current working directory.", default='', type=str)
    parser.add_argument("-t", "--type", default="window", help="[window|delay] The signal parametrisation type", type=str)
    parser.add_argument("-1", "--adaptor_1", action='store_true', help="Extract keyDown events")
    parser.add_argument("-2", "--adaptor_2", action='store_true', help="Extract keyDown and mouseDown events")
    parser.add_argument("-3", "--adaptor_3", action='store_true', help="Extract keyDown, mouseDown, and fixation events")
    parser.add_argument("-k", "--distinguish_keys", action='store_true', default=False, help="Extract four features (instead of one) for keystrokes: alphanumeric (normal), deletion, control, and navigation.")
    parser.add_argument("-w", "--window_length", default=5000, help="The window length in ms. Only effective if --type = window.", type=int)
    return parser

if __name__ == "__main__":
    # set up argument parser
    parser = getArgumentParser()
    args = vars( parser.parse_args() )
    # make sure target_dir ends in '/'
    source = args['source']
    parametrisation = args['type']
    output_dir = args['output_dir']
    if not output_dir.endswith(os.sep) and output_dir != '':
        output_dir += os.sep
    # select appropriate adaptor
    adaptor = None
    if parametrisation == 'window':
        if args['adaptor_1']:
            adaptor = XMLAdaptorMultiWindow1(window_length=args['window_length'], distinguish_keydowns=args['distinguish_keys'])
        elif args['adaptor_2']:
            adaptor = XMLAdaptorMultiWindow2(window_length=args['window_length'], distinguish_keydowns=args['distinguish_keys'])
        elif args['adaptor_3']:
            adaptor = XMLAdaptorMultiWindow3(window_length=args['window_length'], distinguish_keydowns=args['distinguish_keys'])
        else:
            sys.exit("Error: No adaptor chosen. See --help for more info.")
    elif parametrisation == 'delay':
        if args['adaptor_1']:
            adaptor = XMLAdaptorMultiDelay1(distinguish_keydowns=args['distinguish_keys'])
        elif args['adaptor_2']:
            adaptor = XMLAdaptorMultiDelay1(distinguish_keydowns=args['distinguish_keys'])
        elif args['adaptor_3']:
            adaptor = XMLAdaptorMultiDelay1(distinguish_keydowns=args['distinguish_keys'])
        else:
            sys.exit("Error: No adaptor chosen. See --help for more info.")
    else:
        sys.exit("Invalid parametrisation. --type must be 'window' or 'delay'.")       
    # read source files, convert each file, and serialise it
    for file_path in source:
        observation_sequence = adaptor.convert(file_path)
        output_path = output_dir + os.path.basename(file_path) + '.csv'
        try:
            observation_sequence.save(output_path)
        except AttributeError:
            pass # a warning will be issued in this case