# SFP Converter For DNN Accelerator
提供將訓練好的 DNN 模型由浮點數轉為加速器內部運算的定點數格式 - SFP（Static Floating Point）
* 以訓練資料評估神經元輸出的正規化因子，並以神經元為單位做正規化
* 可自行選擇定點數的最小精度（2^(−1)  ~ 2^(−15)）

使用者需提供訓練好的 DNN 模型（為 hdf5 的格式）、訓練資料與測試資料（格式如下），工具會以訓練資料決定神經元輸出的正規化因子，並以測試資料驗證轉換成定點數後的效果如何

轉換後的 DNN 參數會存成 DNN 加速器的 testbench 可直接讀取的資料格式，也會提供定點數 DNN 對應的輸入與輸出資料，讓使用者能自行驗證

## 資料格式
分類型 - 第一行的整數代表檔案中共有幾筆資料，再來每筆資料有兩行，一行為 DNN 的輸入資料，以倍精度且 16 進制的格式表示，另一行為分類答案
![GITHUB](image/dataformat_classification.png)

回歸型 - 除了第一行依然為代表資料總筆數的整數外，之後的每一行皆表示一筆資料，同樣以倍精度且 16 進制的格式表示
![GITHUB](image/dataformat_regression.png)

## 模型轉換 - hdf5 到 model.txt
```
python3 getModel.py path_to_model input_neurons_number
```

## 使用方法 - 定點數轉換
需提供以下參數
* -p：指定的定點數最小精度
* -m：訓練完的 DNN 模型位置
* -td：訓練資料檔案位置
* -vd：測試資料檔案位置

範例
* 15 代表定點數所能表示的最小精度為 2^(−15)
```
./SFP –p 15 -m “model.txt” -td “train_data.txt” -vd “test_data.txt”
```

## 在 DNN 硬體加速器上執行轉換後的 DNN 模型
1. 先以本工具提供的 model_convert.py 將 hdf5 格式的 DNN 模型轉成本工具可接受的格式
2. 執行轉換
3. 在轉換後將 tool 底下的資料夾 model 內的檔案全部移至原 IP 的資料夾 data 底下
4. 以本工具提供的 dnn_tb.v 取代原 IP 提供的
5. 修改原 IP 底下的資料夾 mem_behav 內的檔案 bus_behav.v 中，分別用來指定 DNN 參數位置和 DNN 輸入位置的參數 param_file_name 與 input_file_name
