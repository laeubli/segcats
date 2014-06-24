#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistical Modelling of Human Trnaslation Processes
# Author: Samuel Läubli (slaubli@inf.ed.ac.uk)

"""
Analysis and cleaning tool for potentially error-prone CASMACAT XML files.
"""

from __future__ import division

import sys, os, glob
from collections import defaultdict
from copy import deepcopy
from lxml import etree  # @UnresolvedImport

class CasmacatXMLIterator ( object ):
    
    def __init__ ( self, analyse=True, repair=False, detailed=True ):
        """
        Choose whether the iterator should analyse and/or repair CASMACAT XML files
        fed in thorugh self.process(). If @param detailed is True, errors are logged
        in detail.
        """
        self._analyse = analyse
        self._repair = repair
        self._detailed = detailed
        # constants
        self._EVENT_TYPES = [ 'beforeCopy',
                              'beforeCut',
                              'beforePaste',
                              # 'configChanged',
                              'decode',
                              'fixation',
                              'initialConfig',
                              'keyDown',
                              'keyUp',
                              'loadingSuggestions',
                              'mementoInvalidate',
                              'mouseClick',
                              'mouseDown',
                              'mouseUp',
                              # 'mouseWheelInvalidate',
                              'resize',
                              'scroll',
                              'segmentClosed',
                              'segmentOpened',
                              'selection',
                              'startSession',
                              # 'statsUpdated',
                              'stopSession',
                              'suggestionChosen',
                              'suggestionsLoaded',
                              # 'text',
                              'tokens',
                              'translated',
                              ]
    
    def process ( self, path ):
        """
        Processes the file(s) specified in @param path. Can contain wildcards,
        e.g., /foo/bar/c*.xml.
        """
        # iterate over files
        for file_path in glob.glob(path):
            # read XML
            with open(file_path, 'r') as xml_file:
                # iterate over XML nodes
                try:
                    # initialise control variables and stats
                    self._init()
                    context = etree.iterparse(xml_file, events=('end',), tag=self._EVENT_TYPES, recover=False)
                    self._fast_iter(context)
                except etree.XMLSyntaxError as e:
                    # initialise control variables and stats
                    self._init()
                    self._xml_is_wellformed = [e.msg]
                    # use lxml's recover function to iterparse ill-formed XML as good as possible
                    xml_file.seek(0) # reset
                    context = etree.iterparse(xml_file, events=('end',), tag=self._EVENT_TYPES, recover=True)
                    self._fast_iter(context)
            # evaluate XML / write log
            self._writeLog(os.path.basename(file_path))
            
    def _init ( self ):
        """
        Initialises all control variables and stats (= the log for a single XML file to be analysed)
        """
        self._init_control_vars()
        self._init_stats()

    def _init_control_vars ( self ):
        """
        True = OK; error message (str) otherwise.
        """
        self._xml_is_wellformed = True
        self._no_duplicate_events = True
        self._no_overlapping_events = True
        self._valid_subsessions = True
    
    def _init_stats ( self ):
        """
        Initialises counters and statistics.
        """
        self._events_by_timestamps = defaultdict( lambda: defaultdict(list) ) # e.g., ['fixation']['1234569452'][<node_element>]; should only contain one element per timestamp as it would make no sense for two events of the same type to start at the same time (e.g., two keyDown events)
    
    def _writeLog ( self, file_name ):
        """
        Evaluates the statistics gathered thorugh parsing a single XML file and
        prints the correspondign log to stdout.
        """
        # helper function
        def log( status, var ):
            sys.stdout.write("\t")
            if var == True:
                sys.stdout.write('OK')
            else:
                if len(var) > 1:
                    sys.stdout.write('%s ERRORS' % len(var))
                else:
                    sys.stdout.write('ERROR')
            sys.stdout.write(":\t%s " % status)
            if self._detailed and var != True:
                for line in var:
                    sys.stdout.write('\n\t\t\t- %s' % line)
            sys.stdout.write('\n')
        print file_name
        # Well-formedness of XML file
        log('Well-formed XML', self._xml_is_wellformed)
        # Duplicate events (same event type and start time)
        self._checkDuplicateEvents()
        log('No duplicate events (same event type and start time)', self._no_duplicate_events)
        # Overlapping events (item of type T overlaps with previous item of type T's duration)
        self._checkOverlappingEvents()
        log('No overlapping events (same event type)', self._no_overlapping_events)
        # Valid subsessions (same number of <startSession> and <stopSession> events)
        self._checkValidSubsessions()
        log('Valid subsessions (same number of <startSession> and <stopSession> events)', self._valid_subsessions)
        
    def _checkDuplicateEvents ( self ):
        """
        Checks whether an XML file contains events with the same type at the same start time.
        """
        duplicate_events = []
        for event_type, timestamps in self._events_by_timestamps.iteritems():
            for timestamp, nodes in timestamps.iteritems():
                if len(nodes) > 1:
                    duplicate_events.append( (event_type, timestamp, [node.get('id') for node in nodes]) )
        # format error message for duplicate events
        if len(duplicate_events) > 0:
            self._no_duplicate_events = []
            for event_type, timestamp, event_ids in duplicate_events:
                self._no_duplicate_events.append('%s %s events at start time %s (event IDs: %s)' % (len(event_ids), event_type, timestamp, event_ids))
    
    def _checkOverlappingEvents ( self ):
        """
        Checks whether an XML file contains events of the same time that overlap in time, i.e.,
        event T (start + duration) > event T+1 (start).
        """
        overlapping_events = []
        for event_type, timestamps in self._events_by_timestamps.iteritems():
            previous_event_end = 0
            for timestamp in sorted(timestamps):
                for node in timestamps[timestamp]:
                    if 'duration' in node.attrib:
                        if previous_event_end > int(node.get('time')):
                            duration_of_overlap = previous_event_end - int(node.get('time'))
                            overlapping_events.append( (event_type, timestamp, node.get('duration'), duration_of_overlap, node.get('id')) )
                        previous_event_end = int(node.get('time')) + int(node.get('duration'))
        # format error message for overlapping events
        if len(overlapping_events) > 0:
            self._no_overlapping_events = []
            for event_type, timestamp, event_duration, duration_of_overlap, event_id in overlapping_events:
                self._no_overlapping_events.append( '%s event at start time %s (duration: %s ms, id: %s) overlaps with previous %s event by %s ms.' %  (event_type, timestamp, event_duration, event_id, event_type, duration_of_overlap) )
    
    def _checkValidSubsessions ( self ):
        """
        Checks whether an XML file contains the same number of <startSession> and <stopSession> events.
        """
        num_start_session = len(self._events_by_timestamps['startSession'])
        num_end_session = len(self._events_by_timestamps['stopSession'])
        if num_start_session != num_end_session:
            self._valid_subsessions = ['Invalid subsession markup: %s <startSession> and %s <stopSession> markers found.' % (num_start_session, num_end_session)]
    
    def _fast_iter ( self, context ):
        """
        Quickly iterates over all elements of an xml tree (lxml).
        See http://www.ibm.com/developerworks/xml/library/x-hiperfparse/.
        """
        for event, elem in context:
            self._processNode(elem)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context
    
    def _processNode ( self, node ):
        """
        Handles all nodes passed through self._fast_iter and delgates
        analysis and cleaning to relevant helper functions.
        """
        self._events_by_timestamps[node.tag][int(node.get('time'))].append(deepcopy(node))
  
# TEST DRIVER
if __name__ == "__main__":
    iterator = CasmacatXMLIterator(detailed=False)
    iterator.process('/Users/sam/Documents/ausbildung/uni/msc_ai/thesis/Data/TPR Raw/CFT13/Translog-II/*.xml')   