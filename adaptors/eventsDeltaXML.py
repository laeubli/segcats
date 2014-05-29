#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

from __future__ import division

from lxml import etree
import os, sys, glob, codecs, re, nltk

'''
Converts a TranslogII-style XML document into a CSV representation of event observations.
Each observation consists of the event type (discrete) as well as the time elapsed since
the last event of every possible event type.

Takes 1..* XML files as arguments and creates one observation file each (.obs) in the
current working directory.
'''

EVENTS = ["beforeCopy","beforeCut","beforePaste","configChanged","decode","initialConfig","keyDown","keyUp","loadingSuggestions","mementoInvalidate","mouseClick","mouseDown","mouseUp","mouseWheelInvalidate","resize","scroll","segmentClosed","segmentOpened","selection","startSession","statsUpdated","stopSession","suggestionChosen","suggestionsLoaded","text","tokens","translated"]

def _fast_iter ( context, func, filestream ):
    '''
    Quickly iterates over all elements of an xml tree (lxml)
    See http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''
    for event, elem in context:
        func(elem, filestream)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

def process_log ( node, filestream ):
    global time_elapsed, initialised, EVENTS
    csv_line = []
    event_name = node.tag
    event_abs_time = int(node.get('time'))
    # initialise absolute time at start of session
    if not initialised:
        for e in EVENTS:
            time_elapsed.append(event_abs_time) # this is a bias though; pretend that all events happen first at the very start of the session.
        initialised = True
    # calculate deltas to all event times
    csv_line.append(event_name)
    for i, time in enumerate(time_elapsed):
        csv_line.append(event_abs_time-time)
    # update timestamp for current event
    time_elapsed[EVENTS.index(event_name)] = event_abs_time
    # write csv line
    filestream.write('\t'.join([str(v) for v in csv_line]) + '\n')

# open all files provided as arguments
for filepath in sys.argv[1:]:
    # get file meta information
    filename = os.path.basename(filepath)
    # process xml file contents
    time_elapsed = []
    initialised = False
    with open(filepath, 'r') as xml_file:
        with open(filename + '.obs', 'w') as target_file:
            # write CSV header
            target_file.write('# EVENT, time elapsed since last ' + ' '.join(EVENTS) + '\n') # feature definitions comment
            target_file.write('D\t' + '\t'.join(['W' for event in EVENTS]) + '\n') # feature definitions; use Weibull distribution for continuous features
            # process log events
            context = etree.iterparse(xml_file, events=('end',), tag=EVENTS)
            _fast_iter(context, process_log, target_file)