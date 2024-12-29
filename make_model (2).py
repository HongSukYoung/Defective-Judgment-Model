# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:57:06 2024

@author: USER
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
import numpy as np
import librosa
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

stu_num = "2019146037"
file_path = 'C:/Users/USER/Desktop/2019146037'

# 1. CSV 데이터 불러오기
test_csv_name = stu_num + 'train_data.csv'
test_csv_path = os.path.join(file_path, test_csv_name)
test_csv = pd.read_csv(test_csv_path, index_col=False)

# 2. WAV 파일 불러오기
wav_dir = os.path.join(file_path, 'train_wav')
# Filter out directories and only include .wav files
wav_files = [file for file in glob(os.path.join(wav_dir, '*.wav')) if os.path.isfile(file)]

# CSV 데이터 전처리 함수
def preprocess_csv(test_csv):
    # 1. 특징과 레이블 분리
    # 'Temp'와 'Current' 열의 데이터를 numpy 배열로 변환
    temp_col = test_csv['Temp'].values  # 'Temp' 열 불러와서 변수 처리
    current_col = test_csv['Current'].values  # 'Current' 열 불러와서 변수처리
    input_csv = np.column_stack((temp_col, current_col))  # 두 열을 합쳐 input_csv 변수 선언

    # 'Error' 열의 데이터를 numpy 배열로 변환
    output_csv = test_csv['Error'].values

    # 2. 데이터 스케일링
    # StandardScaler를 사용하여 데이터의 평균을 0, 표준편차를 1로 변환
    scaler = StandardScaler()
    scaler.fit(input_csv)  # 스케일러 학습
    input_csv = scaler.transform(input_csv)  # 변환된 데이터 생성

    # 3. label one-hot encoding
    # 'Error' 값 one-hot encoding
    label_classes = 2  # 레이블의 클래스 개수
    output_csv_one_hot = np.zeros((output_csv.size, label_classes))  # 0으로 채워진 배열 생성
    for i, label in enumerate(output_csv):
        output_csv_one_hot[i, label] = 1  # 인덱스에 1 할당

    return input_csv, output_csv_one_hot

input_csv, output_csv = preprocess_csv(test_csv)

def preprocess_wav(wav_files):
    # 특징을 저장할 리스트 초기화
    wav_feature = []

    for file in wav_files:
        # 1. 오디오 파일 로드
        # librosa.load: 오디오 데이터를 읽어오고, 샘플링 속도(sr)를 반환
        audio, sr = librosa.load(file, sr=None)

        # 2. MFCC 추출
        # librosa.feature.mfcc : MFCC 추출
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # n_mfcc=13으로 13개 Coefficient

        # 3. MFCC의 평균 계산
        # mfcc.T -> (time_step, n_mfcc) 형식 / 평균을 n_mfcc 형식으로 바꿈
        mfcc_mean = []
        for mfcc_coefficients in mfcc:
            mean_value = np.mean(mfcc_coefficients)  # 각 MFCC 평균 계산
            mfcc_mean.append(mean_value)

        mfcc_mean = np.array(mfcc_mean)

        # 4. 특징 리스트에 mfcc 평균 추가
        wav_feature.append(mfcc_mean)

    # 5. 최종 결과 array 바꿈
    wav_feature_array = np.array(wav_feature)
    return wav_feature_array

input_wav = preprocess_wav(wav_files)

# 3. 데이터셋 나누기
# 20 : 80 비율 / random_state = 42
input_csv_train, input_csv_validation, input_wav_train, input_wav_validation, output_train, output_validation = train_test_split(
    input_csv, input_wav, output_csv, test_size=0.2, random_state=42
)

def make_model(csv_shape, wav_shape):
    # CSV_x : CSV 데이터 처리 과정
    # CSV 데이터 = input 정의
    csv_input = Input(shape=csv_shape, name='csv_input')
    # DNN을 통한 특징 추출 / ReLU 활성화 함수 사용 / L2 정규화 추가
    csv_x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(csv_input)
    # BatchNormalization을 통한 안정화 및 성능 향상
    csv_x = layers.BatchNormalization()(csv_x)
    # Dropout을 통한 과적합 방지
    csv_x = layers.Dropout(0.3)(csv_x)
    # DNN을 추가하여 특징을 더 작게 만듬
    csv_x = layers.Dense(64, activation='relu')(csv_x)
    # BatchNormalization을 통한 안정화 및 성능 향상
    csv_x = layers.BatchNormalization()(csv_x)
    # 시계열 형태로 바꿈
    csv_x = layers.Reshape((1, 64))(csv_x)
    # LSTM 레이어를 통해 시계열 정보를 추출
    csv_x = layers.LSTM(32, return_sequences=False)(csv_x)

    # WAV_x : WAV 데이터 처리 과정
    # WAV 데이터 = input 정의
    wav_input = Input(shape=wav_shape, name='wav_input')
    # 시계열 형태로 바꿈
    wav_x = layers.Reshape((wav_shape[0], 1))(wav_input)

    # Residual Block 정의
    # 입력 x와 두 개의 Conv1D를 통해 특징을 추출 -> skip_connection을 적용
    def residual_block(x, filters):
        skip_connect = x  # 스킵 연결을 위한 원본 입력
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        # 입력과 출력의 skip connection 합성
        return layers.Add()([skip_connect, x])

    # Conv1D를 통한 특징 추출
    wav_x = layers.Conv1D(64, kernel_size=3, activation='relu')(wav_x)
    # MaxPooling1D를 통한 축소 및 다운샘플링
    wav_x = layers.MaxPooling1D(pool_size=2)(wav_x)
    # Residual Block을 통해 더 깊은 특징 추출
    wav_x = residual_block(wav_x, 64)
    # Global Average Pooling 사용
    wav_x = layers.GlobalAveragePooling1D()(wav_x)
    # DNN을 통해 더 높은 수준 특징 축소
    wav_x = layers.Dense(64, activation='relu')(wav_x)

    # CSV와 WAV 합침
    combine_wav_csv = layers.concatenate([csv_x, wav_x])

    # Attention Layer: 결합된 데이터를 가중치로 조정
    # 결합된 데이터에 Attention 메커니즘 적용
    attention_wight = layers.Dense(64, activation='tanh')(combine_wav_csv)  # 특징 가중치 학습
    attention_wight = layers.Dense(1, activation='softmax')(attention_wight)  # 가중치를 확률로 변환
    # Attention 가중치를 결합된 데이터에 적용
    combine_wav_csv = layers.multiply([combine_wav_csv, attention_wight])

    # 최종 출력 부분
    z = layers.Dense(64, activation='relu')(combine_wav_csv)  # DNN을 통한 압축
    z = layers.Dropout(0.4)(z)  # Dropout을 통한 과적합 방지
    z = layers.Dense(16, activation='relu')(z)  # 추가 압축
    # 최종 출력 (2개 클래스에 대한 softmax 확률을 출력함)
    output = layers.Dense(2, activation='softmax', name='output')(z)

    # 모델 생성
    model = models.Model(inputs=[csv_input, wav_input], outputs=output)
    # 모델 컴파일 (Adam / categorical_crossentropy)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 생성
model = make_model(csv_shape=(input_csv.shape[1],), wav_shape=(input_wav.shape[1],))
model.summary()

# 5. 모델 학습
history = model.fit(
    [input_csv_train, input_wav_train], output_train,
    validation_data=([input_csv_validation, input_wav_validation], output_validation),
    epochs= 100,
    batch_size=64
)

# 6. 모델 저장
model.save('2019146037_model.h5')