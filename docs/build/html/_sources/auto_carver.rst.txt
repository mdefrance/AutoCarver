AutoCarver
==========

.. _AutoCarver:

AutoCarver, the automated, fast paced data processing pipeline
--------------------------------------------------------------

.. note::
    
    ``AutoCarver`` takes advantage of vectorized groupby on crosstab to gratly improve the combination selection process. 

.. autoclass:: AutoCarver.AutoCarver
    :members: fit, transform, fit_transform

.. autofunction:: AutoCarver.AutoCarver.summary

AutoCarver saving and loading
-----------------------------

.. autofunction:: AutoCarver.AutoCarver.to_json

.. autofunction:: AutoCarver.load_carver
