# tensorflow-finetune-flickr-style

In this project, we use flickr style dataset to demonstrate finetune in TensorFlow.
The details please refer to the example from the [Caffe website](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)

Thank @ethereon for his code. We modify the network.py from [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) for our use.

Thanks @sergeyk for flickr style dataset collection. We modify flickr.py from [vislab](https://github.com/sergeyk/vislab)


### Download flickr style dataset

```sh
# Download dataset
$ python assemble_data.py images train.txt test.txt 500
```

### Download the pre-trained model

Download link: [here](https://drive.google.com/open?id=0B1TxGXQOCIQQME1peW9USXBDME0)

Or follow the tutorial and extract bvlc_alexnet.npy

https://github.com/ethereon/caffe-tensorflow


### Lauch the finetune process

```sh
$ python finetune.py train.txt test.txt bvlc_alexnet.npy
```

### Finetune result

```sh
// Fine-tuning result
Iter 1280: Testing Accuracy = 0.3250
// From scratch result
Iter 1280: Testing Accuracy = 0.1655
```
