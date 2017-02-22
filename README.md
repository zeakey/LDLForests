# Label Distribution Learning Forest
Project page: <http://zhaok.xyz/ldl-forest>

```bash
git clone https://github.com/zeakey/LDLForest --recursive
cd LDLForest/caffe-ldl
# config your own Makefile.config settings
make -j$(nproc) && make pycaffe && cd ..
# get LDL DataSets
bash data/ldl/get_ldl.sh
python experiments/ldl/run.py
```

**Credit to**: [@zeakey](http://zhaok.xyz)

