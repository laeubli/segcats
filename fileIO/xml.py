#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides classes for writing HMMs to and reading HMMs from XML files.
All XML schema definitions are stored in the /fileIO/xsd
"""
import sys, os
from shared import floatToStr
from lxml import etree  # @UnresolvedImport
from sklearn.hmm import MultinomialHMM, GaussianHMM, GMMHMM # @UnresolvedImport

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


class SingleGaussianHMM_XMLSerialiser ( AbstractXMLSerialiser ):
    """
    Converts the parameters of a SingleGaussianHMM to XML.
    """
    def __init__ ( self, states, means_variances, transition_probabilities, comment=None ):
        """
        Takes all HMM parameters to be serialised as an XML document. This base class deals with HMMs
        @param states (list): the hidden states. Must include 'START' (first item) and 'END' (last item)
        @param means_variances (list): the mean and variance for each state's probability density function
            (PDF). One (mean,variance) tuple of floats per state, with their order corresponding to @param
            states
        @param transition_probabilities (list): a list of lists where the first dimension corresponds to
            the index of the from_state and the second dimension to the index of the to_state, according
            to @param states. The transition probabilities from from_state to to_state are floats.
            Example: transition_probabilities[0][2] = -1.084332453
        @param comment (str): A comment that will be added to the root node, e.g., a quick description
            of the model and/or its purpose.
        """
        # add field variables
        self._states = states
        self._means_variances = means_variances
        self._transition_probabilities = transition_probabilities
        self._comment = comment
        # create xml tree
        self._initRoot()
        self._addStates()
        self._addTransitionProbabilities()
        self._addObservationProbabilities()
    
    def _initRoot ( self ):
        """
        @Override _initRoot() from parent class because a SingleGaussianHMM has exactly one feature
        Initialises the root node of the XML document. 
        """
        # initialise default namespace
        self._initNamespace()
        # create root node of the document
        self._root = etree.Element(self._ns + "HMM", nsmap=self._nsMap)
        self._root.attrib['states'] = str(len(self._states)) # add number of states to root node
        self._root.attrib['features'] = str(1) # a SingleGaussianHMM has exactly one feature
        self._root.attrib['type'] = "SingleGaussianHMM"
        if self._comment:
            self._root.attrib['comment'] = self._comment
    
    def _addObservationProbabilities( self ):
        for from_state_index, state_node in enumerate(self._state_nodes[1:-1], 1): 
        # state indices correspond to first dimension of self._transition_probabilities
        # we deliberately leave out the first (=START) and last (=END) state here since they are non-emitting.
            # add <observations> node to each <state> node
            observations_node = etree.SubElement(state_node, self._ns + "observations")
            # add <continuousFeature> feature to <observations>
            continuousFeature_node = etree.SubElement(observations_node, self._ns + "continuousFeature")
            # add Gaussian with mean and variance to <continuousFeature>
            mean, variance = self._means_variances[from_state_index]
            mean = floatToStr(mean)
            variance = floatToStr(variance)
            Gaussian_node = etree.SubElement(continuousFeature_node, self._ns + "Gaussian")
            Gaussian_node.attrib['mean'] = mean
            Gaussian_node.attrib['variance'] = variance
        
        
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
        @return (list): a list of state names, starting with 'START' and ending with 'END'
        """
        return self._states
    
    def getTransitionProbabilities ( self ):
        """
        Returns the transition matrix of the loaded HMM.
        @return (list): a list of lists where the first dimension corresponds to the index of the 
            from_state and the second dimension to the index of the to_state, according to @param 
            states. The transition probabilities from from_state to to_state are floats. Example: 
            transition_probabilities[0][2] = -1.084332453
        """
        return self._transition_probabilities
    
    ### getObservationProbabilities(), getMeansVariances(), getFeatures() etc. need to be card for
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
            assert(start_node_tag in ['state','startState','endState'])
            self._states.append(state_node.get('name'))
            self._state_nodes.append(state_node)
        # just to double-check...
        assert(len(self._states) > 2) # there must be at least a START, an END, and one emitting state
        assert(self._states[0]) == 'START'
        assert(self._states[-1]) == 'END'
    
    def _readTransitionProbabilities ( self ):
        """
        @pre: _readStates() needs to be executed beforehand.
        Reads all transition probabilities between states and stores them in 
            self._transition_probabilities.
        """
        # initialise transition probability matrix with all zeros as zero-transitions are *not* explicitly mentioned in the XML serialisation
        transition_probabilities = [ [None]*len(self._states) for state in self._states ]
        for state_node in self._state_nodes[:-1]: # leave out the END state as it has no outgoing transitions
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
                transition_probabilities[from_state_index][to_state_index] = float(probability)
        self._transition_probabilities = transition_probabilities
            
            
class SingleGaussianHMM_XMLReader ( AbstractXMLReader ):
    """
    Reads the parameters of a SingleGaussianHMM from an XML file, i.e., loads a HMM model 
    from disk. The XML files must validate against the XSD stored in /fileIO/xsd/hmm.xsd.
    """
    
    def __init__ ( self, path ):
        """
        Reads the XML file at @param path, validates it, and extracts the HMM parameters. Upon
        completion, the HMM parameters can be accessed via
            self.getStates()
            self.getTransitionProbabilities()
            self.getMeansVariances()
        @param path: the location of the XML file to be read
        """
        AbstractXMLReader.__init__(self, path)
        if 'type' in self._root.attrib.keys():
            if self._root.attrib['type'] != 'SingleGaussianHMM':
                sys.stderr.write('Warning: Loading HMM parameter definitions of type "%s", while "SingleGaussianHMM" was expected.' % self._root.attrib['type'])
        self._readMeansVariances()
        
    def getMeansVariances ( self ):
        """
        Returns the means and variances (=observation probabilities) from the loaded XML.
        @return (list): returns the mean and variance for each state's probability density function
            (PDF). One (mean,variance) tuple of floats per state, with their order corresponding to
            self._states
        """
        return self._means_variances
    
    def _readMeansVariances ( self ):
        """
        @pre: the constructor of the parent class must be executed beforehand
        @pre: _readStates() needs to be executed beforehand
        Reads the mean and variance for each state's Gaussian and stores them in 
            self._means_variances.
        """
        means_variances = [(None, None)] * len(self._states) # initialise empty vector for all states
        for state_node in self._state_nodes[1:-1]: # leave out the START and END state as they are non-emitting
            # get index of from_state
            state_name = state_node.get('name')
            state_index = self._states.index(state_name)
            # get <observations> node and make sure there is exactly one <continuousFeature> attached to it
            observations_node = state_node.find( self._ns + "observations")
            for i, feature_node in enumerate(observations_node):
                # make sure the format is correct
                if i > 0 or feature_node.tag != self._ns + 'continuousFeature':
                    sys.exit("Error reading HMM definition from XML file: In a SingleGaussianHMM, every state has exactly one continuousFeature. This condition is not met in the provided XML file.")
                # get Gaussians
                for j, gaussian_node in enumerate(feature_node):
                    # make sure the format is correct
                    if i > 0:
                        sys.exit("Error reading HMM definition from XML file: In a SingleGaussianHMM, each state's continuousFeature has exactly one Gaussian. This condition is not met in the provided XML file.")
                    mean = gaussian_node.get('mean')
                    variance = gaussian_node.get('variance')    
                    # add mean and variance to the respective vector
                    means_variances[state_index] = ( float(mean), float(variance) )
        self._means_variances = means_variances