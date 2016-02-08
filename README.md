# tensorflow-finetune-flickr-style

In this project, we use flickr style dataset to demonstrate finetune in TensorFlow.
The details please refer to the example from the [Caffe website](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)

Thank @ethereon for his code. We modify the network.py from [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) for our use.

### Download flickr style dataset

```sh
# Download dataset in the current folder
$ python assemble_data.py
```

### Download the trained model

Please follow the tutorial and extract caffenet.npy

https://github.com/ethereon/caffe-tensorflow

### Motified dataset path in finetune.py

```sh
# Dataset path
train_list = '/path/to/data/flickr_style/train.txt'
test_list = '/path/to/data/flickr_style/test.txt'
```
### Lauch the finetune process

```sh
$ python finetune.py
```

