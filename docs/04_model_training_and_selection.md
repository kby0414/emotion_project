# 모델 학습과 선정

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
- 이 프로젝트의 역할: 웹캠 FPS와 모델 크기를 중시할 때의 경량 후보

### EfficientNet-B0

EfficientNet은 네트워크 깊이, 너비, 입력 해상도를 균형 있게 확장하는 compound scaling 개념을 사용한다. B0는 계열의 기본 모델이며 MBConv와 채널 중요도를 조정하는 SE 구조를 활용한다.

- 장점: 비교 모델 중 가장 높은 Accuracy와 Macro-F1
- 단점: MobileNetV2보다 파라미터와 연산량이 큼
- 이 프로젝트의 역할: 최종 정확도 중심의 기본 후보

## 2. 선정 이유

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

검증 Macro-F1이 가장 높았던 epoch의 가중치다. 최종 웹캠·추론 프로그램에 전달할 파일은 이것이다. 모델 이름, 클래스 순서, 입력 크기, 정규화 값, 최고 지표와 실행 환경도 함께 들어 있다.

### last.pt

가장 마지막 epoch의 가중치와 optimizer·scheduler·AMP 상태를 포함한다. 중단 학습 재개용이며 최종 배포 모델은 아니다. 프로젝트 정리 과정에서 최종 실험의 `last.pt`는 제거했으므로 현재 보관 모델은 `best.pt`다.

## 8. 최종 결과

| 순위 | 모델 | 완료/Best epoch | Accuracy | Macro-F1 | 파라미터 | 시간 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | 18/15 | 0.7400 | 0.7357 | 4.02M | 233.0분 |
| 2 | MobileNetV2 | 21/18 | 0.7309 | 0.7296 | 2.23M | 153.5분 |
| 3 | ResNet18 | 19/16 | 0.7275 | 0.7253 | 11.18M | 227.9분 |

세 모델 모두 목표 50 epoch 이전에 조기 종료됐다. 이는 실패가 아니라 검증 Macro-F1이 3회 연속 개선되지 않아 과적합과 불필요한 계산을 줄인 결과다.

## 9. 최종 모델 선정

기본 최종 모델은 EfficientNet-B0다.

- Accuracy와 Macro-F1 모두 1위
- 특히 현재 약점 클래스인 상처 F1도 세 모델 중 가장 높음
- 4.02M 파라미터로 ResNet18보다 작음

다만 웹캠 프로그램에서는 정확도뿐 아니라 실제 FPS, 지연 시간, 메모리도 측정해야 한다. EfficientNet-B0가 목표 속도를 만족하지 못하면 MobileNetV2가 현실적인 대안이다.

## 10. 비교의 한계

- 제공 Validation과 Training 사이에 동일 인물·촬영 세션으로 추정되는 식별자 중복 가능성이 있다.
- MobileNetV2만 PyTorch 2.14.0+cu126으로 기록됐고 다른 두 모델은 2.13.0+cu126이다.
- 흑백 입력만 비교했으므로 RGB 입력과의 ablation test는 아직 없다.
- 최종 배포 전 실제 웹캠 영상에서 조명·각도·얼굴 크기에 대한 외부 검증이 필요하다.

