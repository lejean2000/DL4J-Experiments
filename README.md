# FashionMnistClassifier
This code runs a simple model on the Fashion MNIST dataset. It assumes that the files of the dataset (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, and t10k-labels-idx1-ubyte) are downloaded in a folder, which is passed as args[0].
Upon first run the UDX files are unpacked into PNGs automatically.

The program also depends on a configuration file which is passed as args[1]. A sample (called fmnist.cfg) is included in this repo. You can configure there the number of epochs and batch size. If you run on GPU, which is preferable, you need to change your pom.xml for org.nd4j as described in the comment in it.

The network structure itself is in getModel(). You will need to play with this of course, to get adequate accuracy.

It runs in Java 11.