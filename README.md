# 🏞️ CNN Classification  

### 🏃🏻 100가지 스포츠 이미지 분류  
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/4086bf36-ea31-48e9-83d9-4985b1b5d63b' width="800">

📌 데이터셋 주제 
- 100가지 종류의 스포츠 이미지를 EfficentNet & ResNet 사전훈련 모델을 사용하여, 예측합니다.
   
### ✏️ 100가지 스포츠 이미지 분류 프로젝트 작업 목차  
  1. [데이터 불러오기 및 전처리](#전처리-작업)
  2. [DataGenerator 설정](#DataGenerator)
  3. [DataFrame 생성](#DataFrame-생성)
  4. [기존 데이터 내 이미지 시각화](#이미지-시각화)
  5. [EfficentNet 사전훈련 모델 훈련 실시](#1-Cycle)
  6. [예측 결과 시각화](#1-Cycle-Result)
  7. [ResNet 사전 훈련 모델 훈련 실시](#2-Cycle)
  8. [성능 비교](#2-Cycle-Result)
  9. [결론](#Total-Result)

## 전처리 작업
✏️ 해당 데이터세트 에서는 훈련 데이터, 검증 데이터, 테스트 데이터에 관련된 폴더가 따로 있었기 때문에, 그대로 각각의 경로들 불러왔습니다.
<details><summary>👉 전처리 작업 코드 확인</summary>
<br>
<h5>Train, Validation, Test 경로 설정</h5>
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/7fa89645-95c2-486c-80bc-b20c2d5db0c8' width="600px"> 
  
</details>

## DataGenerator  
✏️ 기본 이미지 사이즈는 224 x 224로 설정하였고, Batch Size는 32로 설정하였습니다.    
<details><summary>👉 DataGenerator 설정 코드 확인</summary>

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/eb24f21d-7b83-4762-8bbc-f6347e586f34' width="600px">  

<h5>✏️ 훈련데이터에서는 ImageDataGenerator를 통해 훈련 데이터에 대한 augmentation 설정을 정의하고, flow_from_directory 함수를 사용하여 데이터를 불러옵니다.</h5>  
<h5>✏️ 나머지 검증데이터와 테스트 데이터에서는, augmentation 설정 없이 flow_from_directory 함수를 사용해서, 데이터를 불러옵니다.</h5>  

<h5>📌 이미지 전처리 기능 : 확대/축소, 좌우반전, 가로/세로 이동범위 설정<h5> 
  
</details>

## DataFrame 생성  
✏️ 총 100개의 타겟데이터 클래스 분포 확인 하였고, 훈련,검증,테스트 데이터에 대한 각각의 데이터 프레임을 생성하였습니다.  

<details><summary>👉 DataFrame 생성 코드 확인</summary>

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/9a094ce0-8c50-4988-88f3-7ccc93b3bcfd' width="500px">  

<h5>✏️ Datagenerator 생성 과정에서 나온 Class 분포를 확인하여 100개의 타겟데이터의 클래스 분포를 확인하였습니다.</h5>


<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/bfa4dff6-e4a7-4d1f-bb79-53aed768d447' width='600px'>  

<h5>✏️ `train_generator`의 `class_indices`를 이용해 클래스 인덱스를 클래스 이름으로 매핑하는 딕셔너리 `target_name`을 생성합니다.</h5> 

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/d824e32b-8145-44b2-be58-f53fe0cc31ba' width="600px"> 

<h5>✏️ 학습, 테스트, 검증 데이터셋의 클래스 인덱스를 해당 클래스 이름으로 변환해 각 리스트(`train_target_names`, `test_target_names`, `validation_target_names`)에 저장합니다.</h5>   

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/39b80d24-7357-4016-a9b9-8fbdeca14557' width="600px"> 

<h5>✏️ 파일 경로(`file_paths`), 클래스 이름(`target_names`), 클래스 인덱스(`targets`)를 포함한 훈련, 테스트, 검증 데이터프레임을 생성하고, 파일 경로 형식을 정리합니다. 마지막으로 데이터프레임을 출력합니다.</h5>

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/26d07505-3c16-4f76-b93a-8eb57205ecab' width="600px"> 

<h5>✏️ 출력된 DataFrame 이미지 </h5>
</details>

## 이미지-시각화   
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/07fe35d0-e090-4b8d-bb30-3ee7768c1d7b' width="600">  

✏️ 여러가지 실제 스포츠 이미지를 볼 수 있습니다.  

## 1 Cycle
- 💡 EfficentNetB0 사전훈련 모델 훈련 실시  
<details><summary>💡 EfficentNet이란 ? </summary>  

<img src="https://github.com/dosel70/DeepLearning_Project/assets/143694489/6bb2370f-1b37-49ee-bf79-73a795731c57" width="600px">  


-  EfficientNet은 네트워크의 깊이를 깊게 하고, 채널의 길이(필터 수)를 늘리며, 입력 이미지의 해상도를 높이는 방식으로 모델의 성능을 향상시킨 모델입니다.

-  1. More Width (channel Width increase!)
   2. More Deep (Network Depth more deeper!)
   3. More Resolution Scaling (High-Resolution Image)
> 👦🏻 EfficentNet은 이 3가지의 최적의 조합을 AutoML을 통해 찾은 논문입니다. 그래서 위 조합을 효율적으로 만들 수 있도록 하는 compound scaling 방법을 제안하며 이를 통해 더 작은 크기의 모델로도 높은 성능을 보여준 모델입니다.

<h5> 즉, EfficientNet은 모델의 크기와 복잡도를 동시에 증가시키면서도 효율적으로 성능을 극대화한 최신 딥러닝 모델입니다. </h5>
</details>


<details><summary>👉 사전훈련모델 생성 코드 확인</summary>

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/2cd38815-345c-4ed6-adf4-f79f48f7b6af' width="500px">    

✏️ 층 구조를 설정할 수 있는 create_model이라는 함수 생성  

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/3b204646-0703-401c-a6bb-5b398ced1123' width="500px">  

✏️ 사용할 훈련 모델 선택 및 최적화 방식, 손실 함수 설정  
</details>

<details><summary>👉 사전훈련모델 훈련 및 시각화 코드 확인</summary>  
<br>
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/1d6a08bd-c929-4f05-8fa0-bd3be8542e07' width="500px">    

✏️ Callback API 등록 (Early stopping, educeLROnPlateau)  

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/557e02c7-8715-49cc-ac89-8f8c13dfc4c4' width="500px">  

✏️ 모델 훈련 세팅    

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/a0123e07-d200-4b8a-9a23-1e3c8ce2c1e3' width="500px">  

✏️ 훈련 결과  
  - ✨ 훈련,검증,테스트 데이터 모두 accuracy 성능이 0.96~0.98 가량의 점수가 나왔고, 매우 좋은 성능결과가 나왔습니다. 

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/777db389-fb6c-4559-a48b-144c5d8d99e1' width='500px'>  

✏️ Accuracy, Loss 시각화 그래프  
<h5> 1. 사전훈련모델의 훈련데이터와 검증데이터 간의 손실값과 정확도를 Epoch 횟수별로 시각화하였습니다.</h5>  
<h5> 2. 그 결과 검증데이터 부분에서 epoch 4정도에서 약간 튀는 수치를 보였지만, 최종 epoch에 다다라서는, 훈련 데이터의 결과와 비슷하게 변하는 양상을 볼 수 있습니다. 이를 보았을 때, 과적합의 가능성은 없다고 판단하였습니다. </h5>
</details>

## 1 Cycle Result
- #### ✨ 예측 결과 시각화 (기존 Test 폴더 내 이미지 예측)
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/405a02d3-02d8-463a-bd6d-17f5978c1724' width='600px'>  

- #### ✨ 예측 결과 시각화 (인터넷에서 가져온 스포츠 이미지 예측)  
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/7b681b96-742b-4462-b720-b7240ff0e857' width='600px'>

### 🚩 Result  
- 실제로 모든 스포츠 이미지들이 정확히 예측되는 것을 볼 수 있으며, 심지어 농구공만 보여주었을 뿐인데도, 농구로 예측 하는 것을 볼 수 있습니다.  

> EfficentNet_b0 사전훈련 모델 결과 테스트 데이터의 정확도는 97%, 나머지 훈련 데이터 및 검증 데이터 또한 0.96이상의 값이 나오므로, 매우 성능이 좋은 모델이라고 볼 수 있습니다.
>
> 실제 이미지 Predict 결과 또한 거의 100%에 가까운 예측률을 보이고 있으므로, 매우 좋은 성능을 보입니다.  

## 💡 2 Cycle  
- 💡 ResNet50 사전훈련 모델 훈련 실시   

<details><summary>👉 ResNet 사전훈련모델 생성 코드 확인</summary>

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/2cd38815-345c-4ed6-adf4-f79f48f7b6af' width="500px">    

✏️ 층 구조를 설정할 수 있는 create_model이라는 함수 생성  

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/33d8a33c-33b6-4ab4-8b79-6d8b307aa26b' width="500px">  

✏️ 사용할 훈련 모델 선택 및 최적화 방식, 손실 함수 설정  
</details>  

<details><summary>👉 ResNet 사전훈련모델 훈련 및 시각화 코드 확인</summary>  
<br>
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/1d6a08bd-c929-4f05-8fa0-bd3be8542e07' width="500px">    

✏️ Callback API 등록 (Early stopping, educeLROnPlateau)  

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/557e02c7-8715-49cc-ac89-8f8c13dfc4c4' width="500px">  

✏️ 모델 훈련 세팅    

📌 훈련데이터(acc) : 0.8231 , 검증데이터(acc) : 0.8700   
<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/6496e3c6-9a50-4144-953b-99d17a263351' width="500px">   

📌 테스트데이터(acc) : 0.8874    

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/de3fe816-4fbe-4358-a373-a5161aa79943' width="500px">   
   

✏️ 훈련 결과  
  - ✨ 훈련,검증,테스트 데이터 모두 accuracy 성능이 0.82~0.88 가량의 점수가 나왔고, EfficentNet 사전훈련 모델보다는 낮지만 괜찮은 성능을 보여주었습니다.

<img src='https://github.com/dosel70/DeepLearning_Project/assets/143694489/b8cfa434-75f2-4942-bedb-8ee1becf26b2' width='500px'>  
  
<h3>✨ 2 Cycle Result </h3>  

<h4> ✏️ Accuracy, Loss 시각화 그래프 </h4>  
- 사전훈련모델의 훈련데이터와 검증데이터 간의 손실값과 정확도를 Epoch 횟수별로 시각화하였습니다.  

- 그 결과 ResNet50 사전훈련 모델을 사용하였을 때 전체적으로, 검증데이터와 테스트데이터의 accuracy가 더 높은 것을 알 수 있었고, 훈련 데이터 또한 0.82가량으로 나왔으며, epoch를 더 많이 주었으면, 더 높아질 가능성이 높기때문에, 해당 ResNet50 모델 또한 성능이 좋은 것을 알 수 있습니다.
</details>  

### Total Result  
> 최종 CNN 분석 결과 최적의 성능을 보이는 사전훈련 모델은 97~98 % 정도의 정확도를 보인 EfficentB0 모델이라고 볼 수 있겠습니다.
