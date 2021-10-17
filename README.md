# Workpiece_Positioning

## 摘要
本專案為澄德基金會「[**2021 大專校院機電暨智慧創意實作競賽**](https://www.chengde.org.tw/page.php?menu_id=16&p_id=77)」以及109學年第2學期國立成功大學機械系「[**物聯網與大數據於智慧製造應用**](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0108&sem=2&co_no=E134300&class_code=)」課程之期末專題的主要程式，本文檔將會敘述使用環境以及程式架構。

## 程式概述
本專案之程式主要運行於Nvidia Jetson Nano上，會使用WebCam進行影像擷取，並將擷取的影像校正後進行影像處理，之後使用YOLOv4建立的工件識別模型判別工件種類，同時測量其尺寸以及位置，最終回傳至機台控制端進行補正。

## 實作環境
本專案可大致分為硬體與軟體兩部分，硬體部分將敘述設備之硬體規格，軟體部分將敘述軟體運行之所需環境，以下將就這兩部分進行說明。
### 硬體
本專案之程式主要運行於Nvidia Jetson Nano上，並搭配WebCam進行影像擷取的作業，其硬體規格如下所述。

#### [**Nvidia Jetson Nano 2GB**](https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/)

|項目|版本
|:---:|:---:
|作業系統|Ubuntu 18.04.5 LTS
|CPU|四核心 ARM® A57 @ 1.43 GHz
|GPU|配備 128 個核心的 NVIDIA Maxwell™
|記憶體|4 GB 64 位元 LPDDR4 25.6 GB/秒
|儲存空間|SanDisk Ultra 64GB
|CUDA|10.2
|Python|3.6.9

#### WebCam
WebCam使用ASUS推出的[**Webcam C3**](https://www.asus.com/tw/accessories/streaming-kits/all-series/asus-webcam-c3/)，以流暢的 30 fps 輸出畫質銳利的 FHD (1920 x 1080) 視訊，但因感光能力較差，在正常光源下有過曝情形，故改用Logitech推出的[**C920 PRO**](https://www.logitech.com/zh-tw/products/webcams/c920-pro-hd-webcam.960-001062.html)，其感光能力較佳，且也能以流暢的 30 fps 輸出畫質銳利的 FHD (1920 x 1080) 視訊。

### 軟體
本專案之主程式使用Python程式語言以及搭配Numpy、OpenCV等套件包撰寫。另外建置YOLOv4工件辨識模型於Google Colaboratory中進行建置，並將模型轉換為TensorRT版本，以利調用GPU加速運算。以下將對兩部分之環境進行說明。

#### 主程式

|項目|版本
|:---:|:---:
|Python|3.6.9
|OpenCV|4.5.1
|Numpy|1.19.4
|Paho-mqtt|1.5.1
|TensorRT|8.0.1
|Numba|0.34.0
|Pycuda|2019.1.2

#### Google Colaboratory

|項目|版本
|:---:|:---:
|作業系統|Ubuntu 18.04.5 LTS
|顯示卡|NVIDIA Tesla T4 16GB / NVIDIA Tesla P100 16GB
|CUDA|11.0
|Python|3.7.10
|OpenCV|3.2.0

## 程式流程
1. **MQTT**    
與MQTT Broker建立連線，並設定訂閱之主題，以及其他設定。    

2. **參數計算**    
先計算好一些必要參數，如長度的公制像素比等，以便後續須調用時不用重新計算，節整電腦效能之浪費。

3. **影像擷取**    
使用Webcam進行工件影像擷取，並將擷取到的影像進行校正。

4. **工件辨識**    
將影像輸入建置好的模型進行辨識，將會回傳物件外框、工件類別、以及信心分數。

5. **位置計算**    
基於工件辨識獲得的物件外框進行影像遮罩後，進行輪廓檢測取得工件外觀形狀以及中心位置，並使用公制像素比進行換算以取得實際位置。

6. **回傳**    
使用MQTT將結果回傳至機台控制電腦上進行校正。

## 其他檔案（超過25MB）
因檔案過大無法上傳github，故另外提供雲端空間下載，下載後將檔案放置於[**data資料夾**](https://github.com/vf19961226/Workpiece_Positioning/tree/main/data)中。
* [**YOLOv4權重檔**](http://140.116.86.56:5000/sharing/58yxHTxMn)
* [**YOLOv4權重檔（onnx）**](http://140.116.86.56:5000/sharing/mIB3R7X59)
* [**YOLOv4權重檔（trt）**](http://140.116.86.56:5000/sharing/KPnn1Pqb8)    

## 使用教學
### Nvidia Jetson Nano 設定
1. 取得遠端更新伺服器的套件檔案清單
```
sudo apt-get update
```
2. 安裝更新清單上的更新
```
sudo apt-get -y dist-upgrade
```
3. 清除更新時所下載回來的更新(安裝)檔案
```
sudo apt-get clean
```
4. 自動清除更新後用不到的舊版本檔案
```
sudo apt-get autoremove
```
5. 安裝文字編輯器Vim
```
sudo apt-get install vim
```
6. 安裝pip
```
sudo apt-get install python3-pip
pip3 install --upgrade pip
```
7. 從github下載此程式碼
```
git clone https://github.com/vf19961226/Workpiece_Positioning.git
```
8. 安裝程式所需的套件包
```
cd Workpiece_Positioning
pip install -r requirements.txt
```
### YOLOv4模型訓練
於Google Colaboratory中訓練YOLOv4模型（[**連結**](https://colab.research.google.com/drive/1nP3mpV-nqMOppTdIv8poFU7VAdTEapRK?usp=sharing)）（[**訓練集**](https://drive.google.com/drive/folders/1v8_SBl5XwqvwzncSEE8lBQR3WM4E3zJ8?usp=sharing)），必須使用成大帳號登入（**@gs.ncku.edu.tw）。訓練完成後將模型輸入至Nano上。
### 模型轉換
1. 下載模型轉換程式
```
git clone https://github.com/jkjung-avt/tensorrt_demos.git
```
2. 執行ssd資料夾中的`install_pycuda.sh`
```
cd tensorrt_demos/ssd
./install_pycuda.sh
```
若顯示`nvcc not found`的話則需要手動修改`install_pycuda.sh`
```
vim ./install_pycuda.sh
```
將以下內容加入if與fi之間，並將雲本的內容註解掉
```
#!/bin/bash
#
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu

set -e

if ! which nvcc > /dev/null; then
  #echo "ERROR: nvcc not found"
  #exit
  echo "** Add CUDA stuffs into ~/.bashrc"
  echo >> ${HOME}/.bashrc
  echo "export PAYH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" >> ${HOME}/.bashrc
  echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ${HOME}/.bashrc
fi
```
3. 查看pycuda是否安裝成功
```
pip list | grep pycuda
```
4. 安裝onnx1.4.1
```
sudo apt-get install protobuf-compiler libprotoc-dev
sudo pip install onnx==1.4.1
```
5. 建立相關程式
```
cd tensorrt_demos/plugins
make
```
若nvcc報錯則修改`Makefile`中的`NVCC=nvcc`，修改成`NVCC=/usr/local/cuda/bin/nvcc`
6. 將YOLOv4模型轉換為ONNX模型
```
python yolo_to_onnx.py -m My_yolov4
```
7. 將ONNX模型轉換為TensorRT模型
```
python onnx_to_tensorrt.py -m My_yolov4
```
### 使用主程式
1. 執行[**Camera_Positioning.py**](https://github.com/vf19961226/Workpiece_Positioning/blob/main/Camera_Calibration.py)以取得相機參數
2. 執行[**find_circle.py**](https://github.com/vf19961226/Workpiece_Positioning/blob/main/find_circle.py)以取得定位塊相關參數    
此步驟需調整Gaussian濾波參數、肯尼邊緣檢測參數以及定位塊遮罩等，程式碼如下所示。
```py
blur_gray = cv2.GaussianBlur(gray,(3, 3), 0)
canny = cv2.Canny(blur_gray,150,50)

right_bottom = [img.shape[1] -150, img.shape[0]/2 -20]
left_top = [img.shape[1]/2 +75 , img.shape[0]*0 +110]
right_top = [img.shape[1] -150, img.shape[0]*0 +110]
mid_left_bottom = [img.shape[1]/2 +75, img.shape[0]*0 +135]
mid = [img.shape[1]/2 +145, img.shape[0]*0 +135]
mid_right_bottom = [img.shape[1]/2 +145, img.shape[0]/2 -20]
```
4. 將定位塊相關參數複製至[main.py](https://github.com/vf19961226/Workpiece_Positioning/blob/main/main.py)相對應的區域
5. 執行[main.py](https://github.com/vf19961226/Workpiece_Positioning/blob/main/main.py)，等待MQTT發送指令開始辨識
6. 辨識完成後使用MQTT發布辨識結果
* 另有一控制介面[Workpiece Positioning Contoller](https://github.com/vf19961226/Workpiece_Positioning_Contoller)，可發布指令與接收辨識結果。
