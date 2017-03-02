# Label Distribution Learning Forest
## Publication
Code accompanying the paper **Label Distribution Learning Forest**. [[**arXiv**]](https://arxiv.org/abs/1702.06086)  [[**ProjectPage**]](http://zhaok.xyz/LDLForest)

## How to use
### Clone source and build
```bash
git clone https://github.com/zeakey/LDLForest --recursive
cd LDLForest/caffe-ldl
cp Makefile.config.example Makefile.config
make -j$(nproc) && make pycaffe && cd ..
```

*Please customize your own Makefile.config before `make`*.

### Try out LDL toy data:

1. Download LDL datasets:

    `bash data/ldl/get_ldl.sh`

2. Try out LDL example:

    `python experiments/ldl/run.py`

### Facial age estimation:
1. Download the Morph dataset

2. Make LMDB database for Morph:

    `python tools/morph2lmdb.py --data /path/to/Morph`

3. Start to train:

    `python experiments/alexnet/run.py`

A [live demo](https://github.com/zeakey/LDLForest/blob/master/demo.ipynb).
___

Maintained by [@zeakey](https://github.com/zeakey)
