
###**A list of resources and potentially interesting projects.**

####*This list is updated periodically*

**[with TensorFlow or Theano](#with-tensorflow-or-theano)**

- __[Resources](#resources)__

       - [Application](#application)

       - [Theory](#theory)

       - [Tools](#tools)


- __[Basics](#basics)__


- __[Image Processing](#image-processing)__
      
       - [Higher Resolution](#higher-resolution)

       - [Colour](#colour)

       - [Artsy stuff](#generating-art)

- __[Sound](#sound)__
      
       - [generating sound](#generating-sound)       

- __[Text](#text)__

       - [Sentiment analysis](#sentiment-analysis)

       - [Generating Text](#generating-text)

       - [Analyzing Text](#analyzing-text)

- __[Games](#games)__       

- __[Optimization](#optimization)__

- __[Probabilistic programming](#probabilistic-programming)__

**[Caffe, Torch7 and other frameworks](#Caffe,-Torch7-and-other frameworks)**

- __[Projects](#projects)__
- __[Miscellaneous](#miscellaneous)__

- - -

##__With Tensorflow or Theano__

## Resources

####*Application*

* [Tensorflow 101](https://github.com/sjchoi86/Tensorflow-101) - Basics, MLP, CNN, RNN, word2vec, auto-encoders, tensorboard, etc. Jupyter Notebooks.

* [TensorFlow tutorial and examples for beginners](https://github.com/aymericdamien/TensorFlow-Examples) - Nearest neighbour, regression, MLP, CNN, RNNs, auto-encoders, tensorboard, resources. Also features lots more examples using TFlearn.

* [Parallelizing Neural Network Training with Theano](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch13/ch13.ipynb) - Jupyter notebook. 
    * from [Python Machine Learning by Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book)

* [Francois Chollet's Github repository for Keras - examples](https://github.com/fchollet/keras/tree/master/examples) - Python scripts

* [Tensorflow.org tutorials](https://www.tensorflow.org/versions/r0.10/tutorials/index.html) 

* [Machine Learning models in TensorFlow](https://github.com/tensorflow/models)

* [Yet more Tensorflow Tutorials (code)](https://github.com/pkmital/tensorflow_tutorials) - Basics, regression, CNNs, denoising autoencoder, residual network, variational auto-encoder

* [First contact with TensorFlow](http://www.jorditorres.org/first-contact-with-tensorflow/)

* [Learning Machine Learning with TensorFlow](https://github.com/golbin/TensorFlow-ML-Exercises)

* [Deep Learning Tutorials from LISA lab, University of Montreal](http://deeplearning.net/tutorial/deeplearning.pdf) - Extensive resource. Would fit somewhere between 'application' and 'theory'.

* [Short Keras video tutorial](https://www.youtube.com/watch?v=Tp3SaRbql4k)


####*Theory*

* [TensorFlow paper](http://arxiv.org/abs/1603.04467)

* [Colah's blog](http://colah.github.io/) - About neural networks: simply explained concepts. 

* [Intuitive explanation of CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/?utm_content=buffer227b7&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)

* [Matrix Differential Calculus with Tensors (for Machine Learning)](https://github.com/mtomassoli/papers) - Theory only. File is tensor_diff_calc.pdf. 
(Information Theory for Machine Learning is good too - inftheory.pdf)

* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) - Textbook by Michael Nielsen.

####*Tools*

* [Pretty tensor](https://github.com/google/prettytensor) - High level builder API for TensorFlow

* [TFLearn](https://github.com/tflearn/tflearn) - Higher level API for TensorFlow.

* [Scikit Flow: a simplified interface for TensorFlow](http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/)

* [Convert Caffe models to TensorFlow](https://github.com/ethereon/caffe-tensorflow)

* [TensorDebugger](https://github.com/ericjang/tdb) - Interactive debugging and visualization for TensorFlow.

## Basics

####*Hello world, MNIST and the likes*


* [Tensorflow deep neural network for MNIST](https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/notebook) - Tutorial by some guy on Kaggle. Jupyter notebook.


* [MNIST - Generative Adverserial Networks](https://github.com/openai/improved-gan/tree/master/mnist_svhn_cifar10) - Unsupervised learning. Python/Theano/Lasagne.

## Image Processing

####*higher resolution*

* [Image super-resolution through deep learning (GPU-srez)](https://github.com/david-gpu/srez) - Reconstruct blurry pictures. 

####*colour*

* [Colornet: Neural Network to colorize grayscale images](https://github.com/pavelgonchar/colornet) - Python/TensorFlow.

####*generating art*

* [Deep dream](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py) - Python/Keras 

* [Neural doodle](https://github.com/fchollet/keras/blob/master/examples/neural_doodle.py) - Python/Keras 

* [Generating images with generative adverserial networks](https://github.com/openai/improved-gan/tree/master/imagenet) - Unsupervised learning. Python/Tensorflow
    * [Relevant article on generative models](https://openai.com/blog/generative-models/)

* [iGAN: Interactive Image Generation via Generative Adversarial Networks](https://github.com/junyanz/iGAN) - Python/Theano.


* [Generative algorithms - differential lattice](https://github.com/inconvergent/differential-lattice) - Very cool. Python. 


####*Other*

* [Generate Handwriting](https://github.com/hardmaru/write-rnn-tensorflow) - Python/TensorFlow.

* [Neural Image Caption Generation with Visual Attention](https://github.com/kelvinxu/arctic-captions) - Python/Theano.

* [Visual search based on Google's inception model](https://github.com/AKSHAYUBHAT/VisualSearchServer) - Python/TensorFlow.

* [Temporal Convolutional Networks](https://github.com/colincsl/TemporalConvolutionalNetworks) - Identifying human actions. Python/TensorFlow/Keras.

* [Learning Deep Features for Discriminative Localization](https://github.com/jazzsaxmafia/Weakly_detector) - Localization with CNNs. Python/TensorFlow.

## Sound

#### Generating sound

* [DeepJazz](https://github.com/jisungk/deepjazz) - Generate elevator jazz (LSTM). Python/Theano/Keras.

* [Generate classical music](https://github.com/hexahedria/biaxial-rnn-music-composition) - RNN. Python/Theano.

* [Fast Wavenet: An efficient Wavenet generation implementation](https://github.com/tomlepaine/fast-wavenet) - Python/TensorFlow.

## Text

#### Sentiment Analysis

* [Sentiment classification with Bayesian LSTM regression](https://github.com/yaringal/BayesianRNN/tree/master/Example) - Recurrent neural networks/dropout. Python/Theano/Keras.

#### Generating Text

* [LSTM text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) - RNNs. Python/Keras

* [Neural storyteller](https://github.com/ryankiros/neural-storyteller) - Generate stories about images with RNNs. Python/Theano/Lasagne.

* [Translate Shakespeare in modern English](https://github.com/tokestermw/tensorflow-shakespeare) - Python/TensorFlow.

* [Deep QA, sentence prediction (chatbot)](https://github.com/Conchylicultor/DeepQA) - RNN. Python/TensorFlow.

#### Analyzing text

* [Sentence classification](https://github.com/dennybritz/cnn-text-classification-tf) - CNN. Python/TensorFlow

## Games

* [Deep Q-learning - Replicating Deep Mind](https://github.com/spragunr/deep_q_rl) - Playing Atari with Deep Reinforcement Learning. Python/Theano/Lasagne.
    * Also: [Deep Q-learning Pong](http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html) - Python/TensorFlow.
    * Also: [Asynchronic Q-learning Atari](https://github.com/coreylynch/async-rl) - Python/TensorFlow.
    * Also: [Deep learning flappy bird](https://github.com/yenchenlin/DeepLearningFlappyBird) - Python/TensorFlow.


* [Deep pink, a chess AI](https://github.com/erikbern/deep-pink) - Python/Theano.

* [Partial replication of AlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) - Python/Keras.

## Optimization

* [Learning to learn by gradient descent by gradient descent](https://github.com/deepmind/learning-to-learn) - in Python/TensorFlow.

## Probabilistic programming

* [Neural Networks in PyMC3 estimated with Variational Inference](http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/) - tutorial PyMC3/Theano (Part I)

* [Bridging PyMC3 and Lasagne to build a Hierarchical Neural Network](http://twiecki.github.io/blog/2016/07/05/bayesian-deep-learning/) - tutorial PyMC3/Lasagne/Theano (Part II)

* [Mixture density networks for galaxy distance determination](http://cbonnett.github.io/MDN.html) - Python/TensorFlow.

* [Mixture density networks with Edward, Keras and TensorFlow](http://cbonnett.github.io/MDN_EDWARD_KERAS_TF.html)

* [Gumbel-Softmax Variational Autoencoder with Keras](https://github.com/EderSantana/gumbel) - Jupyter notebooks.

## __Caffe, Torch7 and other frameworks__

#### Projects

* [Generate a TED talk based on previous TED talks](https://github.com/samim23/TED-RNN) - Python/Torch

* [Search and filter videos based on content](https://github.com/agermanidis/thingscoop) - CNN. Python/Caffe.

* [Neuralsnap: generate poetry based on images](https://github.com/rossgoodwin/neuralsnap/tree/master/neuralsnap) - Python/Torch

* [Face recognition](https://github.com/cmusatyalab/openface) - Python/Torch

* [Contour detection for image prediction](https://github.com/s9xie/hed) - CNN. Python (or Matlab)/Caffe.

* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style) - Builds on the neural algorithm of artistic style by Gatys et al. Lua/Torch.

* [Pixel Level Domain Transfer](https://github.com/fxia22/PixelDTGAN) - Image generation. Lua/Torch.

* [BachBot](https://github.com/feynmanliang/bachbot) - LSTM for music generation. Torch.

* [DeepFool](https://github.com/LTS4/DeepFool) - Fool a network with minimal perturbations. Matlab/Caffe.

* [DeepDream Animator](https://github.com/samim23/DeepDreamAnim) - Generate animations based on DeepDream. Python/Caffe.

#### Miscellaneous

* [Torch7 wiki cheatsheet](https://github.com/torch/torch7/wiki/Cheatsheet)


