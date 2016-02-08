# tensorflow-finetune-flickr-style

In this project, we use flickr style dataset to demonstrate finetune in TensorFlow.
Please refer to the example from the Caffe website.

http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html

Also we modify the network.py from repository

https://github.com/ethereon/caffe-tensorflow

### Download flickr style dataset

```sh
# Download dataset in the current folder
$ python assemble_data.py
```

### Download the trained model

Please follow the tutorial from and extract caffenet.npy

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

