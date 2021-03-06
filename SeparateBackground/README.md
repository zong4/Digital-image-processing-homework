# Include

* reportWord.docx
* config.py
* another_model.py
* train.py
* test.py
* ./py_path/model.py
* ./py_path/dataset.py
* ./py_path/norm.py
* ./dataset_path/
* ./model_path/

# Run

1. 下载该文件夹。

2. 将[数据集](https://pan.baidu.com/union/challenge/dataset?competition_id=3&code=1715396826)下载至 **dataset_path** 并解压。

3. 先运行 **train.py** ，再运行 **test.py** 。

4. 保险起见，可以先去 **./py_path/dataset.py** 修改相关路径。

5. 同时为了获得更好的效果，可以调用 **GPU** 训练，以及在 **config.py** 中增加训练图片的数量（也可以使用自己的）。

6. 如果有任何问题，可以发起 **issue** 。

---

1. Download this folder.

2. Download [the dataset](https://pan.baidu.com/union/challenge/dataset?competition_id=3&code=1715396826) to **dataset_path** and unzip it.

3. Run **train.py** and then run **test.py**.

4. To be on the safe side, modify the relevant path in **./py_path/dataset.py** firstly.

5. Also, for better results, you can call the **GPU** training and increase the number of training images in **config.py** (or use your own).

6. If you have any questions, you can open **issue** .

# Result

This only use **2000 images** to train.

![](result.png)

# Understand

可以看看 **reportWord.docx** 以及[心得](https://zong4.github.io/2022/06/01/SeparateBackground/)。

You can read [this paper](https://arxiv.org/abs/2108.07009)。
