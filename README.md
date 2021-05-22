# Find_Workpiece_Positioning

## 摘要
本專案為澄德基金會「[**2021 大專校院機電暨智慧創意實作競賽**](https://www.chengde.org.tw/page.php?menu_id=16&p_id=77)」以及109學年第2學期國立成功大學機械系「[**物聯網與大數據於智慧製造應用**](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0108&sem=2&co_no=E134300&class_code=)」課程之期末專題的主要程式，本文檔將會敘述使用環境以及程式架構。

## 程式概述
本專案之程式主要運行於Nvidia Jetson Nano 2GB上，會使用2部WebCam進行影像擷取，並將擷取的影像校正後進行影像處理，之後使用YOLOv4建立的工件識別模型判別工件種類，同時測量其尺寸以及位置，最終回傳至機台控制端進行補正。

## 實作環境
本專案可大致分為硬體與軟體兩部分，硬體部分將敘述設備之硬體規格，軟體部分將敘述軟體運行之所需環境，以下將就這兩部分進行說明。
### 硬體
本專案之程式主要運行於Nvidia Jetson Nano 2GB上，並搭配2部WebCam進行影像擷取的作業，其硬體規格如下所述。

#### Nvidia Jetson Nano 2GB

|項目|版本
|:---:|:---:
|作業系統|Ubuntu 18.04.5 LTS
|顯示卡|
|記憶體|
|CUDA|10.2
|Python|3.7.10

#### WebCam
WebCam使用ASUS推出的

### 軟體
本專案之主程式使用Python程式語言以及搭配Numpy、OpenCV等套件包撰寫。另外建置YOLOv4工件辨識模型於Google Colaboratory中進行建置。以下將對兩部分之環境進行說明。

#### 主程式

|項目|版本
|:---:|:---:
|Python|3.7.10
|OpenCV|4.5.1
|Numpy|1.19.4
|Paho-mqtt|1.5.1

#### Google Colaboratory

|項目|版本
|:---:|:---:
|作業系統|Ubuntu 18.04.5 LTS
|顯示卡|NVIDIA Tesla T4 16GB / NVIDIA Tesla P100 16GB
|CUDA|11.0
|Python|3.7.10
|OpenCV|3.2.0

## 程式流程


## 其他檔案（超過25MB）
[**YOLOv4權重檔**](https://drive.google.com/file/d/1faaJZJvF5MQV_GsRJ9hcQIa_ofdKvUCO/view?usp=sharing)    
此檔案放置位置為data資料夾中    
sudo wget "https://drive.google.com/u/0/uc?id=1faaJZJvF5MQV_GsRJ9hcQIa_ofdKvUCO&export=download" "My_yolov4_best.weights"

## 使用教學
1. sudo apt-get update
2. sudo apt-get -y dist-upgrade
3. sudo apt-get clean
4. sudo apt-get autoremove
5. 將路徑移至桌面
6. sudo mkdir Workpiece_Positioning
7. 移至剛剛創建的資料夾
8. git clone https://github.com/vf19961226/Workpiece_Positioning.git
9. 如果已經下載過則用 git pull https://github.com/vf19961226/Workpiece_Positioning.git
10. 移至剛剛下載的github資料夾內
11. sudo pip install -r requirements.txt

## 安裝git
sudo apt-get install git-all
