Segmenting the image using D-Wave's Qbsolv
===========================================

Two python scripts are available for running the model on qbsolv:

1 - 'qbsolv_mrf_python.py' : Uses built-in python library for qbsolv
2 - 'qbsolv_mrf.py' : Uses cmd build of qbsolv

You may need to know that according to experiments made cmd build works significantly faster compared to built-in python library.


1 - CMD Build:
------------------------

_(Note that to run it image input must be inserted in 'img' folder in root directory of the project.)_

Afterwards following code snippet can be used to run algorithm on image desired:

`python qbsolv.py 'img_name'`

Example run:

`python qbsolv.py 1_small.png`

2 - Built-in Python Library:
------------------------

_(Note that to run it image input must be inserted in 'img' folder in root directory of the project.)_

Afterwards following code snippet can be used to run algorithm on image desired:

`python qbsolv_python.py 'img_name'`

Example run:

`python qbsolv_python.py 1_small.png`