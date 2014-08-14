segcats
=======

Learns statistical models of human translation processes (HTPs) from keylogging and eye tracking data from recorded translation sessions, and uses these models to segment recorded translation sessions into HTP phases.

The modelling approach is described in detail in

Läubli, Samuel. 2014. Statistical Modelling of Human Translation Processes.
MSc Thesis, School of Informatics, The University of Edinburgh.

```bibtex
@mastersthesis{laubli2014msc,
	Author = {L\"{a}ubli, Samuel},
	Title = {Statistical Modelling of Human Translation Processes},
	School = {School of Informatics, The University of Edinburgh},
	Address = {Edinburgh, UK},
	Year = {2014}}
```

### Requirements / Installation

segcats depends on the following frameworks and libraries, which you'll need to install before running any of the modules described below:

* Python 2.7.x
* NumPy > 1.8 (see http://www.numpy.org/)
* lxml (see http://lxml.de)
* Scikit-learn version 0.14 (see http://scikit-learn.org/0.14/)

After downloading your copy of segcats, you'll also need to add it to your PYTHONPATH:

```
export PYTHONPATH=$PYTHONPATH:/your/path/to/the/base_directory/of/segcats
```


### Feature Extraction

The feature extraction process is described in detail in Section 4.1 of the thesis referenced above. To convert recorded CasMaCat XML Logs (CFT14 format) into sequences of feature vectors, use `extract.py`. Example:

```bash
python extract.py -w 500 -3 -k -o sample_data/training_extracted/ sample_data/training/*.xml
```

parametrises all recorded translation sessions in ```sample_data/training_extracted/``` into time windows of length ```-w```=500ms, using adaptor 3 (```-3```), i.e., counts of keystrokes, mouse clicks, and eye tracking data. Run ```python extract.py --help``` for more information.


### Model Training

The model learning and parameter optimisation procedure is described in detail in Section 4.2 of the thesis referenced above. To train a model on the parametrised translation sessions obtained through the above example, run

```bash
python train.py -k 6 -m 10 -o model.xml training_extracted/*.csv
```

This trains a GMM-HMM model with ```-k```=6 hidden states (HTPs) and ```-m```=10 Gaussian mixture components per hidden state. Run ```python train.py --help``` for more information.


### Decoding (Tagging)

To add the most likely hidden state to each observation of a parametrised translation session (see above), run

```
python decode.py -m model.xml training_extracted/P07_P1.xml.csv P07_P1.tagged.csv
```

where ```training_extracted/P07_P1.xml.csv``` is the session to be tagged, and ```P07_P1.tagged.csv``` is the file that the tagged version of this session should be written to.

### Visualisation of Tagged Translation Sessions

The output of ```decode.py``` can be visualised using [viscats](https://github.com/laeubli/viscats).

### Further Information

For further information, please have a look at the class documentation of the individual modules, and feel free to contact me at slaubli ät inf dot ed dot ac dot uk for questions and/or feedback.
