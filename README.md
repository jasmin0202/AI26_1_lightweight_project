# 🚀 TinyImageNet Classification with Knowledge Distillation & SAM

AI 경량화 이미지 분류 프로젝트입니다. 파라미터 수 제한(5,000,000 이하) 조건 시스템에서 모델 체급 한계를 극복하기 위해 **지식 증강(Knowledge Distillation)** 및 **SAM(Sharpness-Aware Minimization)** 옵티마이저를 도입하여 모델의 일반화 성능을 극대화한 전 과정을 기록했습니다.

---

## 📌 1. Project Overview & Constraints
* **Dataset**: TinyImageNet (200 Classes, 100,000 Training images)
* **Core Constraint**: **Strictly under 5,000,000 (5M) Total Parameters**
* **Target**: 경량화 모델 환경에서 Teacher-Student 학습을 통한 Top-1 Accuracy 극대화 및 오버피팅 제어.

---

## 🛠️ 2. Model Architectures (Lightweight Networks)
제한된 파라미터 예산 내에서 성능을 최대로 끌어올리기 위해 대표적인 경량 네트워크 2종을 커스텀하여 앙상블 파이프라인을 구축했습니다.
* **ShuffleNetV2**: 채널 셔플링을 통한 연산 효율성 극대화 및 파라미터 세이브.
* **MobileNetV3 Large**: Inverted Residual Block 및 SE(Squeeze-and-Excitation) 구조 활용.

---

## 📈 3. Experimental Roadmap & Core Logs (실험 및 트러블슈팅 연대기)

### 🔹 Phase 1: Baseline Training
* 표준 AdamW 옵티마이저와 Cross-Entropy Loss 기반의 초기 학습 진행.
* 경량 모델 특성상 표현력(Capacity) 한계로 인해 일정 수준 이상의 성능 정체 직면.

### 🔹 Phase 2: Knowledge Distillation (지식 증강 적용)
* **Teacher Model**: 사전 학습된 대형 **ResNet50** (Top-1 Accuracy: 70.48%) 활용.
* **Student Models**: ShuffleNetV2 & MobileNetV3 Large
* **Strategy**: Soft Label에에 대한 KL-Divergence Loss와 Hard Label에 대한 Cross-Entropy Loss를 결합하여 학습 지휘.
* **Result**: **단일 Student 모델들의 성능이 Baseline 대비 약 1% 유의미하게 상승함.** 스승의 정밀한 확률 분포(Softmax Logits)를 흡수하여 소형 모델의 일반화 능력이 개선됨을 확인.

### 🔹 Phase 3: The Ensemble Dilemma (앙상블 스코어 교란 현상 트러블슈팅) ⭐
* **Problem**: KD를 통해 단일 성능이 1% 올랐음에도 불구, 두 모델을 Soft Voting 방식으로 앙상블하여 제출했을 때 **오히려 최종 리더보드 점수가 꼬꾸라지는 현상** 발생.
* **Analysis**: 
  1. **오답 동기화(Overlapping Errors)**: 동일한 Teacher(ResNet50) 밑에서 배운 두 Student 모델이 정답뿐만 아니라 스승의 '오답 족보(Bias)'마저 똑같이 닮아버림. 집단지성이 발현되지 않고 특정 오답에 대한 확신이 동기화됨.
  2. **확신 과잉 (Over-confidence/Logit Scale Mismatch)**: 경량 모델이 강하게 KD 학습을 거치면서 특정 클래스에 대해 95%~99% 수준의 극단적이고 뾰족한 소프트맥스 확률을 출력. 단순 평균(Average Pooling) 시 멍청한 오답을 내는 모델의 똥고집(99% 확신)이 정답 모델의 확률을 뭉개버리는 참사 발생.
* **Action**: 무지성 앙상블을 배제하고, 수학적 가중치를 부여한 Weighted Ensemble 및 Logit Calibration(온도 스케일링) 튜닝 실험을 거쳐 최종 단일 최고 존엄 모델의 Private 스코어 방어 전략 수립.

### 🔹 Phase 4: Sharpness-Aware Minimization (SAM) Optimizer 
* 앙상블 교란을 극복하고 단일 모델 자체의 강건함(Robustness)을 극대화하기 위해 Loss Landscape을 평탄하게 만들어주는 **SAM 옵티마이저** 이식.
* 가중치 공간에서 손실이 가장 급격하게 변하는 섭동(Perturbation)을 찾아 이를 억제하는 방식으로 학습 진행.
* **Result**: 최적화 평면의 극점을 무디게(Flat Minima) 만듦으로써, 로컬 Validation 셋과 Hidden Test 셋 간의 성능 격차(Drop Rate)를 기존 **0.90%에서 0.54% 수준으로 감소**시키며 일반화 성능 치트키임을 증명.

---

## 🧑‍💻 4. Repository Structure
```text
├── models/
│   ├── shufflenet_v2_custom.py  # 커스텀 셔플넷 구조 코드
│   └── mobilenet_v3_custom.py   # 커스텀 모바일넷 구조 코드
├── train_kd.py                  # Knowledge Distillation 학습 파이프라인
├── train_sam.py                 # SAM 옵티마이저 기반 가중치 튜닝 코드
├── ensemble.py                  # Weighted Ensemble 및 Logit 분석 스크립트
├── .gitignore                   # 대용량 가중치(*.pt, *.pth) 및 데이터셋 보안 제외
└── README.md                    # 본 프로젝트 보고서
