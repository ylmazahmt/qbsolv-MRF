MRF Model that is used for algorithms sites here
================================================

Current Model:
--------------

# 1 - Unary Potential:

_Brighter pixels are more likely to be labelled as foreground according to intensity_
_Darker pixels are more likely to be labelled as background according to intensity_

# 2 - Doubleton Potential:

NULL for now.

TO DO:
------

_Superpixels will be nodes for MRF model._
_Law's masks will be applied to superpixels_
_Distances will be calculated between histograms of superpixels_

# 1 - Unary Potential:

_A hardcoded foreground and background model for each image will be kept._

# 2 - Doubleton Potential:

_Between neigboring superpixels._
