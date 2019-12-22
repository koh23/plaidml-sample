# 脱NVIDIAを夢見て ~ PlaidML ~

## 前書き

機械学習で必ずと言っていいほど必要なデバイスといえば、GPUです。データが少ない場合やモデルが単純な場合などでは、CPUでも処理することが現実的に可能ですが、データの量が多くなると、CPUだけではモデルを訓練するのに時間がかかりすぎるため、GPUを使用します。  
しかし、機械学習に用いるGPUはNVIDIAのものが主流です。理由としては、GPU上で計算する環境: CUDAが使えるのはNVIDIAのGPUだけだからです。  
機械学習で遊ぶ環境としてGoogleのColabがあり、GPUを使用することができますが、Jupyter notebookからしか使えないので、取り回しが悪いです。  

手元のマシンで機械学習で遊べたらいいなと思い、調べてみたところ[PlaidML](https://github.com/plaidml/plaidml)というフレームワークを見つけました。PlaidMLは、NVIDIAではないGPUがある環境でもKerasなどの機械学習の訓練・予測をGPU上で動かすことができるというフレームワークです。  
このフレームワークを使うにあたり、気になっている点は3つありましたのでそれぞれ調べてみました。  

1. まともに動作するのか?
2. 学習は早く終わるか?
3. 推論は早く終わるか?

## 実験環境

* OS: macOS Catalina
* Hardware: MacBook Pro(13-inch, Late2016)
* CPU: Intel Core i5 2.9GHz
* Memory: 16GB
* GPU: Intel Iris Graphics 550 1536MB

## 結果

さきに結論を書きましょう。  
PlaidMLはまともに動かないばかりか、遅いという結果になりました。  
気になっていた点の結果は下記です。  

1. まともに動作するのか?
    * 学習が正常に進まない
2. 学習は早く終わるか?
    * CPUの方が早い
        * CPU: 110sec程度(1エポックあたり)
        * GPU: 250sec程度(1エポックあたり)
3. 推論は早く終わるか?
    * CPU: 7.7sec程度
    * GPU: 8.4sec程度

機械学習環境としてはしょぼい環境なので、推論くらいは早く終わったらいいなと思っていました。  
私のコードが間違っている可能性もありますが、PlaidMLに関してはもう少し様子をみた方が良いようです。  

## 何が起きたか

### 学習が正常に進まない

実験に使用したのはCifar10のデータであり、モデルは下記のものでした。  
[Kerasのサンプルコード](https://keras.io/examples/cifar10_cnn/)を改変したものです。
ソースコードは下記です。(Functional API使ってます。かっこつけですw)

```python
def generate_model():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model
```

モデルの表示

```txt
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               2097664   
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 2,168,362
Trainable params: 2,168,362
Non-trainable params: 0
```

#### 学習の進捗(Tensorflow(CPU))

CPUで訓練を行ったものは、下記のとおりになった。
accuracyがまだ高くなりそうであり、未学習であると言えるが、学習は正しく進んでいるようだ。  

```txt
Train on 45000 samples, validate on 5000 samples
Epoch 1/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.9332 - accuracy: 0.2931 - val_loss: 1.6271 - val_accuracy: 0.4202
Epoch 2/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.5688 - accuracy: 0.4300 - val_loss: 1.3978 - val_accuracy: 0.4976
Epoch 3/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.3954 - accuracy: 0.4969 - val_loss: 1.2657 - val_accuracy: 0.5520
Epoch 4/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.3010 - accuracy: 0.5349 - val_loss: 1.1652 - val_accuracy: 0.5912
Epoch 5/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.2169 - accuracy: 0.5626 - val_loss: 1.1315 - val_accuracy: 0.6046
Epoch 6/10
45000/45000 [====================] - 114s 3ms/sample - loss: 1.1471 - accuracy: 0.5925 - val_loss: 1.0516 - val_accuracy: 0.6394
Epoch 7/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.0798 - accuracy: 0.6155 - val_loss: 0.9873 - val_accuracy: 0.6566
Epoch 8/10
45000/45000 [====================] - 107s 2ms/sample - loss: 1.0291 - accuracy: 0.6348 - val_loss: 0.9542 - val_accuracy: 0.6672
Epoch 9/10
45000/45000 [====================] - 107s 2ms/sample - loss: 0.9881 - accuracy: 0.6524 - val_loss: 0.8987 - val_accuracy: 0.6926
Epoch 10/10
45000/45000 [====================] - 107s 2ms/sample - loss: 0.9430 - accuracy: 0.6693 - val_loss: 0.8599 - val_accuracy: 0.7030
```

#### 学習の進捗(PlaidML)

PlaidMLでも同じモデルで実験しました。  
下記のように学習が全く進まず、0.1程度(学習せず、ランダムに予測した場合と同等)となりました。  
Lossの値が返ってこないのも不思議です。  
また、1エポックあたり、2.5倍の時間がかかっているのは不思議です。  
GPUとCPUの間で通信が発生しすぎたとしても時間がかかりすぎに思えます。  

```txt
Epoch 1/10

45000/45000 [==============================] - 259s 6ms/step - loss: nan - acc: 0.0996 - val_loss: nan - val_acc: 0.0986
Epoch 2/10

45000/45000 [==============================] - 259s 6ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 3/10

45000/45000 [==============================] - 244s 5ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 4/10

45000/45000 [==============================] - 250s 6ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 5/10

45000/45000 [==============================] - 250s 6ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 6/10

45000/45000 [==============================] - 250s 6ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 7/10

45000/45000 [==============================] - 248s 6ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 8/10

45000/45000 [==============================] - 243s 5ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 9/10

45000/45000 [==============================] - 245s 5ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
Epoch 10/10

45000/45000 [==============================] - 245s 5ms/step - loss: nan - acc: 0.1002 - val_loss: nan - val_acc: 0.0986
```

### 学習だけではなく、推論も遅い

そもそも、推論結果がよくないのに推論速度を試すのはナンセンスかもしれませんが、一応比べます。
5回推論をし、その平均を計算しました。  

* CPU
    ```txt
    Evaluate: elasped time is 7.676613092422485
    predict_elapsed time:  [7.69336987 7.54927683 7.86849618 7.75214696 7.67661309]
    average:  7.707980585098267
    std:  0.10401942095874825
    ```
* GPU
    ```txt
    Evaluate: elasped time is 8.435039043426514
    predict_elapsed time:  [8.66443014 8.38799095 8.35081911 8.405128   8.43503904]
    average:  8.448681449890136
    std:  0.11125726040192159
    ```

## うまく動かない原因の心当たり

PlaidMLとTensorflow用の違いはインポートする関数とKerasのバックエンドを指定する環境変数だけです。  
cnn.pyはPlaidML用のスクリプト、cnn-tensorflow.pyはTensorflow(CPu)のスクリプトです。  

```txt
% diff cnn.py cnn-tensorflow.py 
9,11c9,11
< from keras.datasets import cifar10
< from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
< from keras.models import Model, Sequential
---
> from tensorflow.keras.datasets import cifar10
> from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
> from tensorflow.keras.models import Model, Sequential
```

Tensorflow(CPU)版のコードの中でkerasからレイヤーを呼ばないのは、エラーを避けるためです。
Tensorflowが2.x系になったことがPlaidMLが正しく動かない原因として考えられそうです。  

## 感想

久しぶりの機械学習だったので動かすまで少し苦労しました。  
PlaidMLはまだ、バージョン0.6.4(201912現在)なのでもう少し様子をみた方が良いかもしれません。  
