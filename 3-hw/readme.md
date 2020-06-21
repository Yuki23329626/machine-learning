# Machine Learning Homework 3  
608410117 資工 沈濃翔  
  
## 0、Environment  
系統環境：  
DISTRIB_ID=Ubuntu  
DISTRIB_RELEASE=18.04  
DISTRIB_CODENAME=bionic  
DISTRIB_DESCRIPTION="Ubuntu 18.04.4 LTS"  
NAME="Ubuntu"  
VERSION="18.04.4 LTS (Bionic Beaver)"  
ID=ubuntu  
ID_LIKE=Debian  
PRETTY_NAME="Ubuntu 18.04.4 LTS"  
VERSION_ID="18.04"  
VERSION_CODENAME=bionic  
UBUNTU_CODENAME=bionic  
  
套件管理工具：  
anaconda  
  
套件：  
python: 3.6  
scipy: 1.1.0  
imageio: 2.8.0  
pytorch: 0.4.1  
torchvision: 0.2.1  
matplotlib: 3.2.1  
  
## 1、HW3_1  
下圖為 step 分別為 0, 50, 100, 200 時的生成圖片  

STEP 0:  
![](https://i.imgur.com/u5Hj1si.png)  

STEP 50:  
![](https://i.imgur.com/dQkrGWV.png)  

STEP 100:  
![](https://i.imgur.com/gPXE8ps.png)  

STEP 200:  
![](https://i.imgur.com/rwR1mvU.png)  

生成影像確實有變得比較清晰，  
不過也許是因為dataset太小或是epoch不夠多，生成的影像以人眼來說還不夠真實  
## 2、HW3_2  
以下會分別顯示出 GAN loss 與 CNN loss 隨訓練次數改變的圖  
  
GAN：  
![](https://i.imgur.com/D3W6O64.png)  

CNN：  
![](https://i.imgur.com/rEYKVrc.png)  
  
此處的 CNN 是以作業2訓練的結果作為比較  
執行 pytorch draw.py 可得到此圖，loss為手動將作業2訓練的結果寫入array中  
  
由於Y軸尺度的關係，CNN看起來可能波動幅度較大，其實起伏幅度都在0.05單位之內  
  
CNN 的training loss在epoch=10左右就差不多持平  
Validation loss 在 epoch=20左右開始變動幅度持平  
  
而GAN網路的generator loss到epoch=100才趨近平穩，且loss很難趨近於0  
Discriminator loss則是一開始就比較接近0，到epoch=50左右收斂  
GAN可能要較CNN難以收斂  
  
## 3、HW3_3  
請執行資料夾內的add_noise.py，  
此程式會讀取訓練好的generator模型參數，並且用作業要求的不同的noise來生成圖片  
訓練完的模型參數path為 './CelebA_DCGAN_results/generator_param.pkl'  
本程式使用到pytorch一些較新的function，請安裝以下版本：  
Pytorch=1.5.0; torchvision=0.6.0  

```bash  
Python add_noise.py
``` 

生成圖片:  
N(0,1):  
看起來較為符合原本訓練結果  
![](https://i.imgur.com/8uuHF6K.png)  
  
N(-10,1):  
發現生成圖片都偏向某一種形式，可能是因為normal distribution設定太過偏頗  
導致生成的圖片趨近一個極端  
![](https://i.imgur.com/G0x3iEk.png)  
   
U(0,1):  
由於並非normal distribution，而是使用uniform distribution，  
所以才導致圖片有這樣的結果  
![](https://i.imgur.com/y2jHYJ0.png)  
  
## 4、HW3_4  
以下是使用外部人臉dataset的結果  
使用的是"Labeled Faces in the Wild" 這個 dataset，一共取用了13233 張照片來訓練  
一樣分別列出STEP為0, 50, 100, 200時候的訓練結果  
使用了lfw_data_preprocess.py對照片做預處理  
然後用pytorch_lfw_DCGAN.py來做訓練  

STEP 0:  
![](https://i.imgur.com/8bKe6E9.png)  
  
STEP 50:  
![](https://i.imgur.com/cwAk00n.png)  
  
STEP 100:  
![](https://i.imgur.com/tCHNxRX.png)  
  
STEP 200:  
![](https://i.imgur.com/MRhHwqO.png)  
  
可以發現，最終產生的圖片較為模糊，可能是因為dataset的數量較小  
也可能是因為背景比較雜亂，人物距離大小不一，導致生成的相片比較模糊  
比較好的方法，可能還是需要增加人臉的dataset  
或是增加epoch的次數  
像是原本的dataset有兩百萬張  
且epoch有到達20000  
做了這些調整可能生成的影像品質才會比較高  
