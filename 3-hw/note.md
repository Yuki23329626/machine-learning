# 筆記

## ipynb 轉成 python

```bash
pip install ipython
pip install nbconvert

filename="the .ipynb file name you want to convert to .py file"
jupyter nbconvert --to script $filename.ipynb
e.g.:
jupyter nbconvert --to script train_shapes.ipynb
```