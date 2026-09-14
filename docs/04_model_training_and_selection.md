# 모델 학습과 활용

## 1. 비교 모델

### ResNet18

ResNet은 입력을 여러 층에 통과시킨 결과 `F(x)`에 원래 입력 `x`를 더하는 잔차 연결을 사용한다. 깊은 신경망의 기울기 소실과 성능 저하 문제를 완화한다.

- 장점: 구조가 단순하고 안정적이며 기준 모델로 설명하기 좋음
- 단점: 세 후보 중 파라미터가 가장 많고 현재 결과도 가장 낮음
- 이 프로젝트의 역할: 신뢰할 수 있는 기준선

### MobileNetV2

MobileNetV2는 depthwise separable convolution, inverted residual, linear bottleneck을 사용해 연산량과 파라미터를 줄인다.

- 장점: 2.23M 파라미터로 가장 작고 실시간·모바일 환경에 유리
- 단점: 최고 정확도는 EfficientNet-B0보다 낮음
- 이 프로젝트의 역할: 웹캠에서 빠른 추론을 제공하는 경량 모델

### EfficientNet-B0

EfficientNet은 네트워크 깊이, 너비, 입력 해상도를 균형 있게 확장하는 compound scaling 개념을 사용한다. B0는 계열의 기본 모델이며 MBConv와 채널 중요도를 조정하는 SE 구조를 활용한다.

- 장점: 비교 모델 중 가장 높은 Accuracy와 Macro-F1
- 단점: MobileNetV2보다 파라미터와 연산량이 큼
- 이 프로젝트의 역할: 정확도와 효율의 균형을 확인하는 모델

## 2. 비교 모델 구성 이유

세 모델은 서로 다른 목적을 대표한다.

| 관점 | 대표 모델 |
|---|---|
| 안정적인 CNN 기준선 | ResNet18 |
| 경량·실시간 효율 | MobileNetV2 |
| 정확도와 효율의 균형 | EfficientNet-B0 |

모두 torchvision의 ImageNet 사전학습 가중치를 사용할 수 있어 동일한 데이터와 학습 코드로 비교하기 쉽다.

## 3. 학습 데이터 구성

- Training: 223,570장
- Validation: 52,118장
- 클래스: 7개
- 입력: 224×224 흑백 PNG를 로더에서 RGB 3채널로 복제
- 검증 방식: 제공된 Validation 사용

`--validation-source auto`는 Training과 Validation의 클래스 구성이 같으면 제공 Validation을 사용하고, 클래스가 누락됐을 때만 Training에서 층화 분리한다. 이번 최종 데이터에는 7개 클래스가 모두 있어 제공 Validation을 사용했다.

Training은 감정별 4개 원천 파트 중 2개 파트만 사용한 결과다. 전체 파트에서 이미지를 무작위 삭제한 표본이 아니며, 시간과 연산 자원을 고려해 파트 단위로 학습 범위를 줄였다.

## 4. 주요 학습 설정

| 항목 | 값 | 목적 |
|---|---:|---|
| 목표 epoch | 50 | 충분한 수렴 기회 제공 |
| batch size | 32 | GPU 메모리와 안정성의 균형 |
| 초기 learning rate | 0.0003 | 전이학습에서 안정적인 시작 값 |
| optimizer | AdamW | 학습률 적응과 weight decay 분리 |
| weight decay | 0.0001 | 과적합 완화 |
| label smoothing | 0.1 | 과도한 확신 완화 |
| scheduler | CosineAnnealingLR | 후반 학습률을 부드럽게 감소 |
| patience | 3 | Macro-F1 미개선 3회 시 조기 종료 |
| seed | 42 | 분리와 난수 재현성 향상 |
| 주 지표 | Validation Macro-F1 | 클래스별 성능을 동일 가중 평가 |

학습 증강은 좌우 반전과 작은 회전·이동·확대만 사용했다. 표정의 의미를 훼손할 수 있는 강한 변형은 피했다.

## 5. 실행 예시

모델 하나씩 실행하면 PC별 결과를 관리하기 쉽다.

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py `
  --models efficientnet_b0 `
  --epochs 50 `
  --patience 3 `
  --batch-size 32 `
  --run-name GPU_efficientnet_b0_e50_p3_b32
```

모델 이름만 `resnet18` 또는 `mobilenet_v2`로 바꾸면 된다.

## 6. 한 epoch의 동작

1. 학습 배치를 GPU로 이동
2. 순전파로 7개 클래스 logits 계산
3. CrossEntropyLoss 계산
4. AMP와 GradScaler로 역전파 및 가중치 갱신
5. 검증에서는 gradient 없이 예측
6. 혼동행렬에서 Accuracy와 클래스별 F1 계산
7. 검증 Macro-F1이 최고면 `best.pt` 저장
8. 현재 학습 상태는 `last.pt`에 저장
9. patience만큼 개선이 없으면 조기 종료

## 7. 체크포인트 차이

### best.pt

검증 Macro-F1이 가장 높았던 epoch의 가중치다. 웹캠 프로그램에는 세 모델 각각의 `best.pt`를 전달한다. 모델 이름, 클래스 순서, 입력 크기, 정규화 값, 최고 지표와 실행 환경도 함께 들어 있다.

### last.pt

가장 마지막 epoch의 가중치와 optimizer·scheduler·AMP 상태를 포함한다. 중단 학습 재개용이며 프로그램 추론에는 사용하지 않는다. 프로젝트 정리 과정에서 각 실험의 `last.pt`는 제거했으므로 현재 보관 모델은 `best.pt`다.

## 8. 학습 결과

| 순위 | 모델 | 완료/Best epoch | Accuracy | Macro-F1 | 파라미터 | 시간 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | 18/15 | 0.7400 | 0.7357 | 4.02M | 233.0분 |
| 2 | MobileNetV2 | 21/18 | 0.7309 | 0.7296 | 2.23M | 153.5분 |
| 3 | ResNet18 | 19/16 | 0.7275 | 0.7253 | 11.18M | 227.9분 |

세 모델 모두 목표 50 epoch 이전에 조기 종료됐다. 이는 실패가 아니라 검증 Macro-F1이 3회 연속 개선되지 않아 과적합과 불필요한 계산을 줄인 결과다.

## 9. 세 모델 프로그램 활용

단일 최종 모델을 고정하지 않고 세 모델의 `best.pt`를 모두 프로그램에 포함한다.

- ResNet18: 안정적인 CNN 기준선으로 결과 차이를 확인
- MobileNetV2: 빠른 추론과 낮은 자원 사용량을 확인
- EfficientNet-B0: 현재 Validation에서 가장 높은 정확도와 Macro-F1을 제공

일반 모드에서는 사용자가 선택한 모델 하나만 실행해 FPS를 확보한다. 비교 모드에서는 같은 얼굴 ROI를 세 모델에 입력하고 감정 예측, 신뢰도, FPS와 지연 시간을 모델별로 표시한다. 세 모델을 매 프레임 모두 실행하면 속도가 낮아질 수 있으므로 비교 모드는 프레임 간격을 두거나 필요할 때만 활성화한다.

## 10. 비교의 한계

- 한 사람당 유사한 이미지가 여러 장 존재한다. 제공 Validation과 Training 사이의 공통 식별자 528개가 동일 인물·촬영 세션을 뜻한다면 감정 분류 성능에 인물 특징을 학습한 효과가 섞였을 수 있다.
- Training은 감정별 4개 파트 중 2개 파트만 사용했으므로 결과를 전체 Training 데이터의 성능으로 해석할 수 없다.
- MobileNetV2만 PyTorch 2.14.0+cu126으로 기록됐고 다른 두 모델은 2.13.0+cu126이다.
- 흑백 입력만 비교했으므로 RGB 입력과의 ablation test는 아직 없다.
- 프로그램 적용 전 실제 웹캠 영상에서 조명·각도·얼굴 크기와 모델별 FPS를 검증해야 한다.
