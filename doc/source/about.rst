=====
About
=====

Tomographic reconstruction creates three-dimensional views of an 
object by combining two-dimensional images taken from multiple 
directions, for example, this is how a CAT (computer-aided tomography) 
scanner generates 3D views of the heart or brain. 

Data collection can be rapid, but the required computations are massive 
and often the beamline staff can be overwhelmed by data that are 
collected far faster than corrections and reconstruction  can be 
performed :cite:`Toby:15`. Further, many common experimental perturbations 
can degrade the quality of tomographs, unless corrections are applied. 

To address the needs for image correction and tomographic reconstruction 
in an instrument independent manner, the TomoPy code was developed
:cite:`Gursoy:14a`, which is a parallelizable high performance 
reconstruction code.

