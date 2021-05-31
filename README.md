# Workpiece_Positioning

## 摘要
本專案為澄德基金會「[**2021 大專校院機電暨智慧創意實作競賽**](https://www.chengde.org.tw/page.php?menu_id=16&p_id=77)」以及109學年第2學期國立成功大學機械系「[**物聯網與大數據於智慧製造應用**](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0108&sem=2&co_no=E134300&class_code=)」課程之期末專題的主要程式，本文檔將會敘述使用環境以及程式架構。

## 程式概述
本專案之程式主要運行於Nvidia Jetson Nano 2GB上，會使用2部WebCam進行影像擷取，並將擷取的影像校正後進行影像處理，之後使用YOLOv4建立的工件識別模型判別工件種類，同時測量其尺寸以及位置，最終回傳至機台控制端進行補正。

## 實作環境
本專案可大致分為硬體與軟體兩部分，硬體部分將敘述設備之硬體規格，軟體部分將敘述軟體運行之所需環境，以下將就這兩部分進行說明。
### 硬體
本專案之程式主要運行於Nvidia Jetson Nano 2GB上，並搭配2部WebCam進行影像擷取的作業，其硬體規格如下所述。

#### [**Nvidia Jetson Nano 2GB**](https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-nano/education-projects/)

|項目|版本
|:---:|:---:
|作業系統|Ubuntu 18.04.5 LTS
|CPU|四核心 ARM® A57 @ 1.43 GHz
|GPU|配備 128 個核心的 NVIDIA Maxwell™
|記憶體|2 GB 64 位元 LPDDR4 25.6 GB/秒
|儲存空間|SanDisk Ultra 64GB
|CUDA|10.2
|Python|3.7.10

#### WebCam
WebCam使用ASUS推出的[**Webcam C3**](https://www.asus.com/tw/accessories/streaming-kits/all-series/asus-webcam-c3/)，以流暢的 30 fps 輸出畫質銳利的 FHD (1920 x 1080) 視訊。

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
1. **MQTT**    
與MQTT Broker建立連線，並設定訂閱之主題，以及其他設定。

2. **參數計算**    
先計算好一些必要參數，如長度以及深度的公制像素比等，以便後續須調用時不用重新計算，節整電腦效能之浪費。

3. **影像擷取**    
使用2部Webcam分別作為左右相機進行工件影像擷取。

4. 影像處理

5. 工件辨識

6. 位置計算

7. 回傳

## 其他檔案（超過25MB）
* [**YOLOv4權重檔**](http://140.116.86.56:5000/sharing/58yxHTxMn)
* [**YOLOv4權重檔（onnx）**](http://140.116.86.56:5000/sharing/mIB3R7X59)
* [**YOLOv4權重檔（trt）**](http://140.116.86.56:5000/sharing/KPnn1Pqb8)    
檔案放置位置為data資料夾中    
sudo wget "http://140.116.86.56:5000/fsdownload/58yxHTxMn/My_yolov4.weights" "My_yolov4.weights"    
sudo wget "http://140.116.86.56:5000/fsdownload/mIB3R7X59/My_yolov4.onnx" "My_yolov4.onnx"    
sudo wget "http://140.116.86.56:5000/fsdownload/KPnn1Pqb8/My_yolov4.trt" "My_yolov4.trt"    

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
