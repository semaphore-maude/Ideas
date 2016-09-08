
###**A list of resources and potentially interesting things to run on the GPU.**

*I only included projects in Python with Tensorflow or Theano  -ML*


- [Basics](#hello-world)

- [Resources](#resources)

- [Image Processing](#image-processing)
      
       - [Higher Resolution](#resolution)

       - [Colour](#colour)

       - [Artsy stuff](#art)

- [Words](#words)

       - [Sentiment analysis](#sentiment)

       - [Generating Text](#text-gen)

- [Variational Inference](#variational-inference)



- - -

## Basics

####*Hello world, MNIST and the likes*


* [Tensorflow deep neural network for MNIST](https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/notebook) - Tutorial by some guy on Kaggle - Jupyter notebook


* [MNIST - Generative Adverserial Networks](https://github.com/openai/improved-gan/tree/master/mnist_svhn_cifar10) - Unsupervised learning. Python/Theano/Lasagne


## Resources

####*Application*

* [Tensorflow 101](https://github.com/sjchoi86/Tensorflow-101) - Tutorials (Jupyter notebooks). Basics, MLP, CNN, RNN, word2vec, auto-encoders, tensorboard, etc. 

* [Parallelizing Neural Network Training with Theano](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch13/ch13.ipynb) - Jupyter notebook. 
    * from [Python Machine Learning by Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book)

* [Francois Chollet's Github for Keras - examples](https://github.com/fchollet/keras/tree/master/examples) - Python scripts

* [Tensorflow.org tutorials](https://www.tensorflow.org/versions/r0.10/tutorials/index.html) 

####*Theory*

* [Colah's blog](http://colah.github.io/) - About neural networks: simply explained concepts. 

* [Matrix Differential Calculus with Tensors (for Machine Learning)](https://github.com/mtomassoli/papers/blob/master/tensor_diff_calc.pdf) - Theory only.

## Image Processing

####*higher resolution*

* [Image super-resolution through deep learning (GPU-srez)](https://github.com/david-gpu/srez) - Reconstruct blurry pictures. Github rep: data + python scripts. 

####*colour*

* [Colornet: Neural Network to colorize grayscale images](https://github.com/pavelgonchar/colornet) - Github rep: data + python/tensorflow scripts.

####*generating art*

* [Deep dream](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py) - Python/Keras 

* [Neural doodle](https://github.com/fchollet/keras/blob/master/examples/neural_doodle.py) - Python/Keras 

* [Generating images with generative adverserial networks](https://github.com/openai/improved-gan/tree/master/imagenet) - Unsupervised learning. Python/Tensorflow
    * [Relevant article on generative models](https://openai.com/blog/generative-models/)



## Words

#### Sentiment Analysis

* [Sentiment classification with Bayesian LSTM regression](https://github.com/yaringal/BayesianRNN/tree/master/Example) - Recurrent neural networks/dropout. Python/Theano/Keras.

#### Generating Text

* [LSTM text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) - RNNs. Python/Keras


## Variational Inference

* [Neural Networks in PyMC3 estimated with Variational Inference](http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/) - tutorial with PyMC3/Theano (Part I)

* [Bridging PyMC3 and Lasagne to build a Hierarchical Neural Network](http://twiecki.github.io/blog/2016/07/05/bayesian-deep-learning/) - tutorial with PyMC3/Lasagne/Theano (Part II)
