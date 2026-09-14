# Emotion Project 문서 목차

이 폴더는 데이터 준비부터 세 모델 체크포인트 전달까지 수행한 작업을 단계별로 정리한다. 처음 프로젝트를 받는 팀원은 아래 순서대로 읽으면 된다.

| 순서 | 문서 | 내용 |
|---:|---|---|
| 1 | [프로젝트 개요](01_project_overview.md) | 목표, 전체 흐름, 폴더와 핵심 산출물 |
| 2 | [Python·PyTorch·GPU 환경](02_environment_and_gpu.md) | Python 3.12.10, CUDA 12.6용 PyTorch, 검증 방법 |
| 3 | [EDA와 전처리](03_eda_and_preprocessing.md) | 데이터 품질, 224×224 흑백 PNG 변환, 실패 데이터 |
| 4 | [모델 학습과 활용](04_model_training_and_selection.md) | 세 모델 구조, 학습 설정, 체크포인트, 프로그램 활용 방식 |
| 5 | [평가와 시각화](05_evaluation_and_visualization.md) | Accuracy, Macro-F1, 혼동행렬, 통합 비교 결과 |
| 6 | [데이터셋 이전](06_dataset_transfer.md) | 대용량 전처리 데이터를 다른 PC로 안전하게 복사하는 방법 |
| 7 | [GitHub 협업](07_github_workflow.md) | 코드 공유, Git LFS 모델 공유, 올리면 안 되는 파일 |
| 8 | [오류 대응](08_troubleshooting.md) | 손상 이미지, 재부팅, CUDA, matplotlib, Git 오류 대응 |
| 9 | [발표 질문과 답변](09_presentation_qa.md) | 발표에서 자주 나올 질문에 대한 답변 예시 |

## 주요 산출물 바로 찾기

- 세 모델 체크포인트: `models/GPU_*/<model>/best.pt`
- 세 모델 비교표: `results/comparisons/all_three_models/comparison_table.csv`
- 발표용 비교 그래프: `results/comparisons/all_three_models/`
- 상세 EDA 근거: `results/preprocessing_eda_report.md`
- 전처리 실행 원본 기록: `dataset/processed/preprocessing_summary.json`(Git 제외)

## 문서의 기준

- 최종 데이터: Training 223,570장, Validation 52,118장, 합계 275,688장
- 감정 클래스: 기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립
- 최종 입력: EXIF 방향이 보정된 224×224 1채널 흑백 PNG
- 모델 비교 기준: 제공 Validation의 최고 Macro-F1
- 현재 Validation 최고 수치: EfficientNet-B0, Accuracy 0.7400, Macro-F1 0.7357
- 프로그램 구성: 세 모델 선택 실행과 비교 실행
