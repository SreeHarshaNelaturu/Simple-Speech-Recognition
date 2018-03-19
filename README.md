# Simple-Speech-Recognition
Simple improvised Speech Recognition model, built on top of DeadSimpleSpeechRecognizer.

Used https://github.com/manashmndl/DeadSimpleSpeechRecognizer for data and reference

Following changes implemented:
<ul>
Increased Dropout to reduce overfitting
Optimizer updated to Adam to ensure faster training.
Added an extra Convolution Layer to increase accuracy
Reduced a Dense layer for faster training.
</ul>

The code was run on Keras with Tensorflow Backend, Python 3.6 on a GTX 960M
