#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides classes for writing HMMs to and reading HMMs from XML files.
All XML schema definitions are stored in the /fileIO/xsd
"""
import sys, os

import numpy as np
from scipy import linalg
from lxml import etree  # @UnresolvedImport
from sklearn.hmm import MultinomialHMM, GaussianHMM, GMMHMM # @UnresolvedImport
from sklearn.mixture import GMM # @UnresolvedImport

from shared import floatToStr

XML_NAMESPACE = 'https://github.com/laeubli/segcats/fileIO/xsd/hmm'

class AbstractXMLSerialiser ( object ):
    """
    Converts the parameters of a HMM to an XML representation. This is
    a way of saving the complete HMM to disk. 
    """
    
    def __init__ ( self, states, features, transition_probabilities, comment=None ):
        """
        Takes all HMM parameters to be serialised as an XML document. This base class deals with HMMs
        @param states (list): the hidden states. Must include 'START' (first item)
        @param features (list): the features, i.e., instances of AbstractFeature subclasses. These
            objects store the feature type and observation probabilities for all states.
        @param transition_probabilities (list): a list of lists where the first dimension corresponds to
            the index of the from_state and the second dimension to the index of the to_state, according
            to @param states. The transition probabilities from from_state to to_state are floats.
            Example: transition_probabilities[0][2] = -1.084332453
        @param comment (str): A comment that will be added to the root node, e.g., a quick description
            of the model and/or its purpose.
        """
        # add field variables
        self._states = states
        self._features = features
        self._transition_probabilities = transition_probabilities
        self._comment = comment
        # create xml tree
        self._initRoot()
        self._addStates()
        self._addTransitionProbabilities()
    
    def _initNamespace ( self ):
        """
        Initialises the default namespace for the xml according to the global XML_NAMESPACE
        variable. The LXML-compatible namespace and the mapping will be stored in self._ns and
        self.nsMap, respectively.
        """
        self._nsMap = {None : XML_NAMESPACE} # the namespace map; default = XML_NAMESPACE, hence no prefix
        self._ns = "{%s}" % XML_NAMESPACE
    
    def _initRoot ( self ):
        """
        Initialises the root node of the XML document. 
        """
        # initialise default namespace
        self._initNamespace()
        # create root node of the document
        self._root = etree.Element(self._ns + "HMM", nsmap=self._nsMap)
        self._root.attrib['states'] = str(len(self._states)-1) # add number of states (without START, hence -1) to root node
        self._root.attrib['features'] = str(len(self._features)) # add number of features to root node
        if self._comment:
            self._root.attrib['comment'] = self._comment
    
    def _addStates ( self ):
        """
        Adds one node per state to the root node.
        Adds references to the state nodes to self._state_nodes
        """
        assert(self._states[0] == 'START')
        self._state_nodes = []
        for state in self._states:
            if state == 'START':
                state_name = 'startState'
            else:
                state_name = 'state'
            state_node = etree.SubElement(self._root, self._ns + state_name)
            state_node.attrib['name'] = state
            self._state_nodes.append(state_node)
    
    def _addTransitionProbabilities ( self ):
        """
        Adds the transition probabilities to the state nodes.
        """
        for from_state_index, state_node in enumerate(self._state_nodes): 
        # state indices correspond to first dimension of self._transition_probabilities
            # add <transitions> node to each <state> node
            transitions_node = etree.SubElement(state_node, self._ns + "transitions")
            for to_state_index, transition_probability in enumerate(self._transition_probabilities[from_state_index]):
                if transition_probability != None:
                    transition_probability = floatToStr(transition_probability)
                    transition_node = etree.SubElement(transitions_node, self._ns + "transition")
                    transition_node.attrib['to'] = self._states[to_state_index]
                    transition_node.attrib['probability'] = transition_probability
    
    def _addObservationProbabilities ( self ):
        """
        Adds the observation probabilities for each feature to the state nodes.
        """
        pass # this is implemented by the respective subclasses
    
    def getXML ( self, pretty_print=True ):
        """
        @param pretty_print (bool): whether or not to optimise the XML for human reading
        @return (str): the XML serialisation of the converted HMM
        """
        return etree.tostring(self._root, pretty_print=pretty_print, encoding='UTF-8', xml_declaration=True)
    
    def saveXML ( self, file_path, pretty_print=True ):
        """
        Saves the serialised model to @param file_path.
        """
        with open(file_path, 'w') as f:
            f.write(self.getXML(pretty_print=pretty_print))


def HMMSerialiser ( model, state_names=None, feature_names=None, comment=None ):
    """
    Returns the corresponding HMMSerialiser object for a Multinomial, Gaussian, or GMM HMM model.
    """
    if isinstance(model, MultinomialHMM):
        return MultinomialHMMSerialiser(model, state_names, feature_names, comment)
    elif isinstance(model, GaussianHMM):
        return GaussianHMMSerialiser(model, state_names, feature_names, comment)
    elif isinstance(model, GMMHMM):
        return GMMHMMSerialiser(model, state_names, feature_names, comment)
    else:
        sys.exit("Cannot serialise HMM model to XML: The model must be of type MultinomialHMM, GaussianHMM, or GMMHMM from the sklearn.hmm package.")


class MultinomialHMMSerialiser ( AbstractXMLSerialiser ):
    """
    #TODO
    """
    pass


class GaussianHMMSerialiser ( AbstractXMLSerialiser ):
    """
    Converts the parameters of a GassianHMM model to XML.
    """

    def __init__ ( self, model, state_names=None, feature_names=None, comment=None):
        assert isinstance(model, GaussianHMM)
        self._model = model
        # set default state names if none provided
        if not state_names:
            state_names = ['START'] + ['H%s' % i for i in range(0, model.n_components)]
        # set default feature names if none provided
        if not feature_names:
            feature_names = ['F%s' % i[0] for i in enumerate(model.means_[0])]
        # convert transition matrix
        transition_probabilities = [ [None] + list(model.startprob_) ] + list([ [None] + list(t) for t in model.transmat_]) # [None] denotes zero-transition-probabilities to the START state
        # initialise base class
        AbstractXMLSerialiser.__init__(self, state_names, feature_names, transition_probabilities, comment)
        # set additional root node attributes
        self._root.attrib['type'] = "Gaussian"
        self._root.attrib['covarianceType'] = 'diagonal' if model._covariance_type == 'diag' else 'full'
        # add observation probabilities
        self._addObservationProbabilities()
    
    def _addObservationProbabilities( self ):
        for state_index, state_node in enumerate(self._state_nodes[1:], 0): 
        # state indices correspond to first dimension of self._transition_probabilities
        # we deliberately leave out the first (=START) state here since it is non-emitting
            # add <observations> node to each <state> node
            observations_node = etree.SubElement(state_node, self._ns + "observations")
            # add <continuousFeature> feature to <observations> for each feature
            for feature_index, feature_name in enumerate(self._features):
                continuousFeature_node = etree.SubElement(observations_node, self._ns + "continuousFeature")
                continuousFeature_node.attrib['name'] = feature_name
                # add Gaussian with mean and variance to <continuousFeature>
                mean = self._model.means_[state_index][feature_index]
                mean = floatToStr(mean)
                variance = self._model.covars_[state_index][feature_index][feature_index]
                variance = floatToStr(variance)
                Gaussian_node = etree.SubElement(continuousFeature_node, self._ns + "Gaussian")
                Gaussian_node.attrib['mean'] = mean
                Gaussian_node.attrib['variance'] = variance
                # add covariances if covariance type is 'full'
                if self._model._covariance_type == 'full':
                    for other_feature_index, other_feature_name in enumerate(self._features):
                        if other_feature_index != feature_index:
                            covariance_node = etree.SubElement(Gaussian_node, self._ns + "covariance")
                            covariance_node.attrib['with'] = other_feature_name
                            covariance_node.text = floatToStr(self._model.covars_[state_index][feature_index][other_feature_index])

class GMMHMMSerialiser ( AbstractXMLSerialiser ):
    """
    Converts the parameters of a GMMHMM model to XML.
    """

    def __init__ ( self, model, state_names=None, feature_names=None, comment=None):
        assert isinstance(model, GMMHMM)
        self._model = model
        # set default state names if none provided
        if not state_names:
            state_names = ['START'] + ['H%s' % i for i in range(0, model.n_components)]
        # set default feature names if none provided
        if not feature_names:
            feature_names = ['F%s' % i[0] for i in enumerate(model.means_[0])]
        # convert transition matrix
        transition_probabilities = [ [None] + list(model.startprob_) ] + list([ [None] + list(t) for t in model.transmat_]) # [None] denotes zero-transition-probabilities to the START state
        # initialise base class
        AbstractXMLSerialiser.__init__(self, state_names, feature_names, transition_probabilities, comment)
        # set additional root node attributes
        self._root.attrib['type'] = "GMM"
        self._root.attrib['mixtureComponents'] = str( len(model.gmms_[0].weights_) )
        self._root.attrib['covarianceType'] = 'diagonal' if model._covariance_type == 'diag' else 'full'
        # add observation probabilities
        self._addObservationProbabilities()
    
    def _addObservationProbabilities( self ):
        for state_index, state_node in enumerate(self._state_nodes[1:], 0): 
        # state indices correspond to first dimension of self._transition_probabilities
        # we deliberately leave out the first (=START) state here since it is non-emitting
            # add <observations> node to each <state> node
            observations_node = etree.SubElement(state_node, self._ns + "observations")
            # add <continuousFeature> feature to <observations> for each feature
            component_weights = self._model.gmms_[state_index].weights_
            for feature_index, feature_name in enumerate(self._features):
                continuousFeature_node = etree.SubElement(observations_node, self._ns + "continuousFeature")
                continuousFeature_node.attrib['name'] = feature_name
                # add <GMM> node to <continuousFeature>
                GMM_node = etree.SubElement(continuousFeature_node, self._ns + "GMM")
                for component_index, component_weight in enumerate(component_weights):
                    # add Gaussian with mean, variance, and weight to <GMM>
                    mean = self._model.gmms_[state_index].means_[component_index][feature_index]
                    mean = floatToStr(mean)
                    variance = self._model.gmms_[state_index].covars_[component_index][feature_index]
                    if self._model._covariance_type == 'full':
                        variance = variance[feature_index]
                    variance = floatToStr(variance)
                    Gaussian_node = etree.SubElement(GMM_node, self._ns + "Gaussian")
                    Gaussian_node.attrib['mean'] = mean
                    Gaussian_node.attrib['variance'] = variance
                    Gaussian_node.attrib['weight'] = floatToStr( component_weight )
                    # add covariances if covariance type is 'full'
                    if self._model._covariance_type == 'full':
                        for other_feature_index, other_feature_name in enumerate(self._features):
                            if other_feature_index != feature_index:
                                covariance_node = etree.SubElement(Gaussian_node, self._ns + "covariance")
                                covariance_node.attrib['with'] = other_feature_name
                                covariance = self._model.gmms_[state_index].covars_[component_index][feature_index][other_feature_index]
                                covariance_node.text = floatToStr(covariance)


class AbstractXMLReader ( object ):
    """
    Reads the parameters of a HMM from an XML file, i.e., loads a HMM model from disk.
    The XML files must validate against the XSD stored in /fileIO/xsd/hmm.xsd.
    """
    
    def __init__ ( self, path ):
        """
        Reads the XML file at @param path, validates it, and extracts the HMM parameters. Upon
        completion, the HMM parameters can be accessed via
            self.getStates()
            self.getTransitionProbabilities()
            ... [further parameters depend on the subclass used]
        @param path: the location of the XML file to be read
        """
        # initialise namespace
        self._initNamespace()
        # read and validate the XML file
        self._root = self._loadXML(path)
        # extract the HMM parameters
        self._readStates()
        self._readTransitionProbabilities()
    
    def getStates ( self ):
        """
        Returns the states of the loaded HMM.
        @return (list): a list of state names, without START.
        """
        return self._states[1:] # leave out the START state
    
    def getStateTransitionProbabilities ( self ):
        """
        Returns the transition matrix for transitions between the hidden states of the loaded HMM.
        @return (list): a list of lists where the first dimension corresponds to the index of the 
            from_state and the second dimension to the index of the to_state, according to 
            self._states. The transition probabilities from from_state to to_state are floats. 
            Example: transition_probabilities[0][2] = -1.084332453
        """
        return self._state_transition_probabilities
    
    def getStartTransitionProbabilities ( self ):
        """
        Returns the transition matrix for transitions between the START and any hidden state of
            the loaded HMM.
        @return (list): a list of lists where the first dimension corresponds to the index of the 
            to_state, according to self._states. The transition probabilities from from_state to to_state are floats. Example: 
            transition_probabilities[0][2] = -1.084332453
        """
        return self._start_transition_probabilities
    
    ### getObservationProbabilities(), getMeansVariances(), getFeatures() etc. need to be cared for
    ### in the respective subclass.
    
    def _initNamespace ( self ):
        """
        Initialises the default namespace for the xml according to the global XML_NAMESPACE
        variable. The LXML-compatible namespace and the mapping will be stored in self._ns and
        self.nsMap, respectively.
        """
        self._nsMap = {None : XML_NAMESPACE} # the namespace map; default = XML_NAMESPACE, hence no prefix
        self._ns = "{%s}" % XML_NAMESPACE
    
    def _removeNS ( self, nodeName ):
        """
        @pre: self._initNamespace() needs to be run beforehand.
        Removes the namespace prefix from a node name, e.g.,
        {https://github.com/laeubli/segcats/fileIO/xsd/hmm}startState => startState
        @return (str): nodeName without the namespace prefix
        """
        return nodeName.replace(self._ns, '')
    
    def _loadXML ( self, path ):
        """
        Loads and validates the XML file.
        @return (xml_node): the parsed XML document; the process will abort if the XML is not
            well-formed or valid according to the HMM schema definition.
        """
        # load schema definition
        base_path = os.path.dirname(__file__) # this module's path
        xsd_path = os.path.join( base_path, 'xsd', 'hmm.xsd' ) # the XML schema definition, relative th this module's path
        xsd = etree.XMLSchema( etree.parse(xsd_path) )
        # load XML file
        xml = etree.parse( path )
        # validate XML file
        if not xsd.validate(xml):
            error = xsd.error_log.last_error
            sys.exit("Cannot load HMM parameters from %s as it does not validate against the HMM schema definition.\n%s" % (path, error))
        return xml.getroot()
    
    def _readStates ( self ):
        """
        Reads all state names and stores them in self._states.
        Additionally, references to the <state> XML nodes will be stored in self._state_nodes.
        """
        self._states = []
        self._state_nodes = []
        for state_node in self._root:
            start_node_tag = self._removeNS(state_node.tag)
            assert(start_node_tag in ['state','startState'])
            self._states.append(state_node.get('name'))
            self._state_nodes.append(state_node)
        # just to double-check...
        assert(len(self._states) > 1) # there must be at least a START, an END, and one emitting state
        assert(self._states[0]) == 'START'
    
    def _readTransitionProbabilities ( self ):
        """
        @pre: _readStates() needs to be executed beforehand.
        Reads all transition probabilities between states and stores them in 
            self._transition_probabilities.
        """
        # initialise transition probability matrix with all zeros as zero-transitions are *not* explicitly mentioned in the XML serialisation
        transition_probabilities = [ [None]*len(self._states) for state in self._states ]
        for state_node in self._state_nodes:
            # get index of from_state
            from_state_name = state_node.get('name')
            from_state_index = self._states.index(from_state_name)
            # get <transitions> node and iterate over all <transition> nodes
            transitions_node = state_node.find( self._ns + "transitions")
            for transition_node in transitions_node:
                # double-check
                assert(self._removeNS(transition_node.tag) == 'transition')
                # get index of to_state and transition probability
                to_state_name = transition_node.get('to')
                to_state_index = self._states.index(to_state_name)
                probability = transition_node.get('probability')
                # add transition probability to matrix
                transition_probabilities[from_state_index][to_state_index] = np.float64(probability)
        self._start_transition_probabilities = np.array(transition_probabilities[0][1:])
        self._state_transition_probabilities = np.array([ t[1:] for t in transition_probabilities[1:] ])


class HMMReader ( AbstractXMLReader ):
    """
    Reads the parameters of a Multinomial, Gaussian, or GMM HMM from an XML file, i.e., loads a
    HMM model from disk. The XML files must validate against the XSD stored in /fileIO/xsd/hmm.xsd.
    """
    
    def __init__ ( self, path ):
        """
        Reads the XML file at @param path, validates it, and extracts the HMM parameters. Upon
        completion, the HMM model or individual parameters can be accessed via
            self.getModel()
            self.getStates()
            self.getFeatures()
            self.getStartTransitionProbabilities()
            self.getStateTransitionProbabilities()
        @param path: the location of the XML file to be read
        """
        AbstractXMLReader.__init__(self, path)
        # read additional root attributes and determine model type (Multinomial, Gaussian, GMM)
        self._type = self._root.attrib['type']
        self._num_features = int( self._root.attrib['features'] )
        if self._type in ['Gaussian', 'GMM']:
            self._covariance_type = 'diag' if self._root.attrib['covarianceType'] == 'diagonal' else self._root.attrib['covarianceType']
            if self._type == 'GMM':
                self._num_mixture_components = int( self._root.attrib['mixtureComponents'] )
        self._readObservationProbabilities()
    
    def getModel ( self ):
        """
        Returns a HMM model of type sklearn.hmm.{Multinomial,Gaussian,GMM}, according
        to the loaded XML.
        """
        if self._type == 'Multinomial':
            pass #TODO
        elif self._type == 'Gaussian':
            model = GaussianHMM( n_components = len(self._states[1:]), # exclude START state
                                covariance_type = self._covariance_type,
                                startprob = self.getStartTransitionProbabilities(),
                                transmat = self.getStateTransitionProbabilities(),
                                params = '',
                                init_params = ''
                               )
            model.means_ = self._means
            if self._covariance_type == 'diag':
                model.covars_ = [np.diag(cv) for cv in self._covars]
            else:
                #for n, cv in enumerate(self._covars):
                #    print n, np.allclose(cv, cv.T), np.any(linalg.eigvalsh(cv) <= 0)
                model.covars_ = self._covars # TODO: solve problem with covariance matrices with all equal values
            return model
        elif self._type == 'GMM':
            model = GMMHMM( n_components = len(self._states[1:]), # exclude START state
                                covariance_type = self._covariance_type,
                                startprob = self.getStartTransitionProbabilities(),
                                transmat = self.getStateTransitionProbabilities(),
                                gmms = self._GMMs,
                                params = '',
                                init_params = ''
                               )
            return model

    def _readObservationProbabilities ( self ):
        """
        @pre: the constructor of the parent class must be executed beforehand
        @pre: _readStates() needs to be executed beforehand
        Reads the observation probabilities; this is delegated to the relevant helper function
            for the adequate model type (Multinomial, Gaussian, HMM).
        """
        if self._type == 'Multinomial':
            self._readObservationProbabilitiesMultinomial()
        elif self._type == 'Gaussian':
            self._readObservationProbabilitiesGaussian()
        elif self._type == 'GMM':
            self._readObservationProbabilitiesGMM()
        else:
            sys.exit("Cannot read HMM model from XML: Only models of type Multinomial, Gaussian, or GMM are supproted.")
    
    def _readObservationProbabilitiesMultinomial ( self ):
        """
        TODO
        """
        pass

    def _readObservationProbabilitiesGaussian ( self ):
        """
        Reads the means and covariance matrices for the Gaussians of a Gaussian HMM.
        """
        # initialise empty matrices for means and covariances
        means = [ [ 0.0 for f in range(self._num_features) ] for s in self._states[1:] ] # START state is non-emitting
        covars = [ [ [ 0.0 for f in range(self._num_features) ] for f in range(self._num_features) ] for s in self._states[1:] ] # START state is non-emitting
        self._features = []
        # read observation probabilities for each state
        for state_node in self._state_nodes[1:]: # leave out the START state as it is non-emitting
            # get index of from_state
            state_name = state_node.get('name')
            state_index = self._states[1:].index(state_name)
            # get <observations> node and make sure there is exactly one <continuousFeature> attached to it
            observations_node = state_node.find( self._ns + "observations")
            for feature_node in observations_node:
                # get feature name and index
                feature_name = feature_node.get('name')
                if len(self._features) < self._num_features:
                    self._features.append(feature_name)
                feature_index = self._features.index(feature_name)
                # get Gaussians
                for gaussian_index, gaussian_node in enumerate(feature_node):
                    # make sure the format is correct
                    if gaussian_index > 0:
                        sys.exit("Error reading HMM definition from XML file: In a Gaussian HMM, each state's continuousFeatures have exactly one Gaussian each. This condition is not met in the provided XML file.")
                    mean = np.float64( gaussian_node.get('mean') )
                    variance = np.float64( gaussian_node.get('variance') )
                    # add mean and variance to the respective vector
                    means[state_index][feature_index] = mean
                    covars[state_index][feature_index][feature_index] = variance
                    # read covariances if applicable
                    if self._covariance_type == 'full':
                        for covariance_node in gaussian_node:
                            assert(covariance_node.tag == self._ns + 'covariance')
                            other_feature_name = covariance_node.get('with')
                            if len(self._features) < self._num_features:
                                self._features.append(other_feature_name)
                            other_feature_index = self._features.index(other_feature_name)
                            covariance = np.float64( covariance_node.text.strip() )
                            covars[state_index][feature_index][other_feature_index] = covariance
        self._means = np.array(means)
        self._covars = np.array(covars)
    
    def _readObservationProbabilitiesGMM ( self ):
        """
        Reads the means and covariance matrices for the GMMs of a GMM HMM.
        """
        self._features = []
        self._GMMs = []
        # read observation probabilities for each state (one GMM per state)
        for state_node in self._state_nodes[1:]: # leave out the START state as it is non-emitting
            # initialise empty GMM matrices
            means = [ [ 0.0 for f in range(self._num_features) ] for s in range(self._num_mixture_components) ] # START state is non-emitting
            covars = [ [ [ 0.0 for f in range(self._num_features) ] for f in range(self._num_features) ] for s in range(self._num_mixture_components) ] # START state is non-emitting
            weights = [ 0.0 for w in range(self._num_mixture_components) ]
            # get index of from_state
            state_name = state_node.get('name')
            state_index = self._states[1:].index(state_name)
            # get <observations> node and make sure there is exactly one <continuousFeature> attached to it
            observations_node = state_node.find( self._ns + "observations")
            for feature_node in observations_node:
                # get feature name and index
                feature_name = feature_node.get('name')
                if len(self._features) < self._num_features:
                    self._features.append(feature_name)
                feature_index = self._features.index(feature_name)
                # get Gaussians
                for GMM_index, GMM_node in enumerate(feature_node):
                    for gaussian_index, gaussian_node in enumerate(GMM_node):
                        # get mean, variance, and weight for each Gaussian
                        mean = np.float64( gaussian_node.get('mean') )
                        variance = np.float64( gaussian_node.get('variance') )
                        weight = np.float64( gaussian_node.get('weight') )
                        # store mean, variance, and weight
                        means[gaussian_index][feature_index] = mean
                        covars[gaussian_index][feature_index][feature_index] = variance
                        weights[gaussian_index] = weight # Note: no sanity check here; must sum to 1 and be the same for all GMMs across states
                        # read covariances if applicable
                        if self._covariance_type == 'full':
                            for covariance_node in gaussian_node:
                                assert(covariance_node.tag == self._ns + 'covariance')
                                other_feature_name = covariance_node.get('with')
                                if len(self._features) < self._num_features:
                                    self._features.append(other_feature_name)
                                other_feature_index = self._features.index(other_feature_name)
                                covariance = np.float64( covariance_node.text.strip() )
                                covars[gaussian_index][feature_index][other_feature_index] = covariance
            new_GMM = GMM( n_components = self._num_mixture_components,
                                    covariance_type = self._covariance_type,
                                    params = '',
                                    init_params = '' )
            new_GMM.means_ = np.array(means)
            if self._covariance_type == 'diag':
                new_GMM.covars_ = np.array( [np.diag(np.array(cv)) for cv in covars] )
            else:
                new_GMM.covars_ = np.array(covars)
            new_GMM.weights_ = np.array(weights)
            self._GMMs.append(new_GMM)