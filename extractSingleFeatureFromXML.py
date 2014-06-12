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
    parser = argparse.ArgumentParser(description="Extracts observations of a single event type (e.g., fixation) from 1..* XML files. The observations consist of a single feature, making them suitable to train SingleGaussianHMMs. The type of feature to be extracted can be specified by choosing a suitable adaptor (run this program with --help for more information).")
    parser.add_argument("source", help="The XML file(s) from which to extract observations. Multiple files can be passed via wildcards, e.g., 'P*.xml.' (paths using wildcards must be encapsulated in single quotes).", type=str)
    parser.add_argument("event_type", help="The event type to be considered, e.g., 'fixation'.", type=str)
    parser.add_argument("-o", "--output_dir", help="The target folder where the observation sequences (one CSV file per input XML file) will be stored. The default is the current working directory.", default='', type=str)
    parser.add_argument("-a", "--adaptor_a", action='store_true', help="Event-based extraction of the duration attribute of a single event type. Example: Extract the length of all fixation events.")
    parser.add_argument("-b", "--adaptor_b", action='store_true', help="Event-based extraction of the inter-event delay between any two successive events of a single event type. Note: At the beginning the session (and each subsession if applicable), the first event is not recorded since it obviously has no predecessor to be compared with.")
    parser.add_argument("-c", "--adaptor_c", action='store_true', help="Window-based extraction of the duration attribute of a single event type. Example: Extract the total duration of eye-fixations in windows of equal length. If an event's start and ent time falls between window boundaries, it's duration is split accordingly.")
    parser.add_argument("-d", "--adaptor_d", action='store_true', help="Window-based counting of all events of a single event type. Example: Count the number of fixation events in windows of equal length. The start time decides which window an event belongs to. The first window starts at the start time of the first event.")
    return parser

if __name__ == "__main__":
    # set up argument parser
    parser = getArgumentParser()
    args = vars( parser.parse_args() )
    # make sure target_dir ends in '/'
    source = args['source']
    source = source.strip("'").strip('"') # remove quotes at beginning and end
    event_type = args['event_type']
    output_dir = args['output_dir']
    if not output_dir.endswith(os.sep) and output_dir != '':
        output_dir += os.sep
    # select appropriate adaptor
    adaptor = None
    if args['adaptor_a']:
        adaptor = XMLAdaptorSingleEventA(event_type)
    elif args['adaptor_b']:
        adaptor = XMLAdaptorSingleEventB(event_type)
    elif args['adaptor_c']:
        adaptor = XMLAdaptorSingleEventC(event_type)
    elif args['adaptor_d']:
        adaptor = XMLAdaptorSingleEventD(event_type)
    else:
        sys.exit("Error: No adaptor chosen. See --help for more info.")
    # read source files, convert each file, and serialise it
    for file_path in glob.glob(source):
        observation_sequence = adaptor.convert(file_path)
        output_path = output_dir + os.path.basename(file_path) + '.csv'
        serialiseObservationSequence(observation_sequence, output_path, ['value'])