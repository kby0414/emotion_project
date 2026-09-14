# 평가와 시각화

## 1. 평가 지표

### Accuracy

전체 Validation 이미지 중 정답을 맞힌 비율이다. 전체적인 예측 성공률을 직관적으로 보여 준다.

### Precision

특정 감정으로 예측한 이미지 중 실제로 그 감정인 비율이다. 거짓 양성 예측이 많으면 낮아진다.

### Recall

실제 특정 감정 이미지 중 모델이 해당 감정을 찾아낸 비율이다. 놓친 표정이 많으면 낮아진다.

### F1-score

Precision과 Recall의 조화평균이다. 둘 중 하나만 높은 모델보다 균형 있게 예측하는 모델을 평가한다.

### Macro-F1

7개 감정의 F1을 같은 비중으로 평균한다. 데이터 수가 많은 감정만 잘 맞혀도 높은 점수를 받는 문제를 줄이므로 최종 모델 선정의 주 지표로 사용했다.

## 2. 최종 정량 결과

| 모델 | Accuracy | Macro-F1 | 차이 |
|---|---:|---:|---|
| EfficientNet-B0 | 0.7400 | 0.7357 | 1위 |
| MobileNetV2 | 0.7309 | 0.7296 | 1위 대비 Macro-F1 -0.0062 |
| ResNet18 | 0.7275 | 0.7253 | 1위 대비 Macro-F1 -0.0104 |

## 3. 클래스별 F1

아래 값은 소수 둘째 자리로 반올림한 값이다.

| 감정 | EfficientNet-B0 | MobileNetV2 | ResNet18 |
|---|---:|---:|---:|
| 기쁨 | 0.95 | 0.95 | 0.95 |
| 당황 | 0.77 | 0.76 | 0.76 |
| 분노 | 0.77 | 0.76 | 0.76 |
| 불안 | 0.57 | 0.56 | 0.57 |
| 상처 | 0.54 | 0.53 | 0.50 |
| 슬픔 | 0.72 | 0.71 | 0.71 |
| 중립 | 0.83 | 0.83 | 0.83 |

기쁨과 중립은 비교적 잘 구분하지만, 불안과 상처는 세 모델 모두 어려워한다. 이 두 감정은 표정 특징이 서로 비슷하거나 다른 부정 감정과 경계가 모호할 가능성이 있다.

## 4. 생성되는 시각화

| 파일 | 해석 |
|---|---|
| `training_curves_*.png` | epoch별 학습·검증 Loss, Accuracy, Macro-F1, 학습률 |
| `confusion_matrix_*.png` | 실제 감정이 어떤 감정으로 잘못 예측됐는지 |
| `model_comparison.png` | 모델별 최고 Accuracy와 Macro-F1 |
| `per_class_f1.png` | 모델별 7개 감정 F1 |
| `comparison_table.csv` | 발표표 작성에 사용할 정확한 수치 |
| `presentation_summary.md` | 간단한 최종 순위표 |

혼동행렬은 각 실제 클래스 행의 합이 1이 되도록 정규화한다. 클래스 수가 달라도 색상을 공정하게 비교하기 위해서다.

## 5. 그래프 재생성

```powershell
.\.venv\Scripts\python.exe .\src\visualize_results.py `
  .\results\experiments\GPU_resnet18_e50_p3_b32 `
  .\results\experiments\GPU_mobilenet_v2_e50_p3_b32 `
  .\results\experiments\GPU_efficientnet_b0_e50_p3_b32 `
  --output-dir .\results\comparisons\all_three_models
```

`history.json`에는 epoch별 수치와 혼동행렬이, `run_summary.json`에는 최고 epoch·시간·환경이 들어 있다. 시각화 코드는 마지막 epoch가 아니라 Validation Macro-F1이 가장 높은 epoch를 찾아 비교한다.

## 6. 발표 해석 순서

1. 세 모델이 같은 Training과 Validation으로 평가됐다고 설명한다.
2. Accuracy와 Macro-F1을 함께 보여 준다.
3. EfficientNet-B0가 두 지표 모두 가장 높다고 말한다.
4. 클래스별 F1에서 불안·상처가 공통 약점임을 제시한다.
5. EfficientNet-B0를 최종 후보로 선정하되 실시간 속도는 MobileNetV2와 추가 비교한다고 결론낸다.

## 7. 과장하면 안 되는 부분

- Accuracy 0.7400은 모든 환경에서 74%를 보장한다는 뜻이 아니라 현재 Validation에서의 결과다.
- 세 모델 차이가 크지 않아 다른 seed나 프레임워크 버전에서 순위가 달라질 수 있다.
- 분할 간 식별자 중복 가능성 때문에 완전히 새로운 사람에 대한 성능은 별도 group split이나 외부 데이터로 검증해야 한다.

