# Simple-Speech-Recognition
Simple improvised Speech Recognition model, built on top of DeadSimpleSpeechRecognizer.

Used https://github.com/manashmndl/DeadSimpleSpeechRecognizer for data and reference

Following changes implemented:
<ul>
  <li>Increased Dropout to reduce overfitting</li>
  <li>Optimizer updated to Adam to ensure faster training.</li>
  <li>Added an extra Convolution Layer to increase accuracy</li>
  <li>Reduced a Dense layer for faster training.</li>
</ul>

The code was run on Keras with Tensorflow Backend, Python 3.6 on a GTX 960M
