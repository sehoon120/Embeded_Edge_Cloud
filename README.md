# 📦 Embedded_Edge_Cloud  
**에지-클라우드 협업 기반 고정밀 얼굴 인식 보안 시스템**

---

## 📌 프로젝트 소개

본 프로젝트는 에지 디바이스(Raspberry Pi)와 클라우드 서버(AWS EC2)를 연동하여,  
**고정밀 얼굴 인식 기반의 보안 시스템**을 구현하는 것을 목표로 합니다.  

- 에지에서 얼굴 검출 및 전처리 수행  
- 클라우드에서 고정밀 얼굴 인식 모델을 활용한 신원 확인  
- 전송 지연 최소화 및 에너지 효율 극대화를 위한 시스템 구성  

---

## 🔧 시스템 구성

```
[Raspberry Pi + CSI Camera]
        │
  얼굴 검출 및 전처리 (0.1초 간격)
        │
        ▼
  [HTTP] 전송 (3 frame spike)
        │
        ▼
[AWS EC2 서버 (Python + Flask)]
  - InsightFace 얼굴 인식 추론
  - 응답 반환 (bbox, accuracy, name)
```

---

## 🔗 AWS EC2 서버 환경

- 인스턴스 타입: `t2.micro` (Free-tier)
- GPU 필요 시: `g4dn.xlarge` (NVIDIA T4, 약 $0.526/hr)

---

## 🔗 노트북 서버 환경

- Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 2050

---

## 📁 프로젝트 폴더 구조

```bash
Embeded_Edge_Cloud/
├── Sehoon                     # Sehoon 개별 WorkSpace
├── Jaeyeong                   # Jaeyeong 개별 WorkSpace
├── registered_faces           # 임베딩 등록용 사진 경로
├── edge_only                  # 에지 only 시스템 실행 코드
├── edge_only_performance      # 에지 only 시스템 LFW 성능 테스트 코드
├── README.md                  # 프로젝트 설명 문서
└── progress_1                 # 테스트 결과 이미지 저장 폴더
```

---

## 👥 팀원별 작업 분담

| 이름            | 역할                                  |
| ------------- | ----------------------------------- |
| 세훈 (Sehoon)   | EC2 서버 세팅, 에지+클라우드 모델 최적화 |
| 재영 (Jaeyoung) | 노트북 서버 세팅, 에지+클라우드 모델 최적화 |

---

## ✅ 주차별 계획 및 달성 현황

| 주차 | 계획                             | 달성 여부 | 비고                         |
|------|----------------------------------|-----------|------------------------------|
| 9주차 | EC2 서버 설정, Flask 서버 구성           | ✅        | 테스트 완료                  |
| 10주차| 얼굴 인식 모델 적용 (InsightFace)     | ✅        | 기본 임베딩 기반 확인           |
| 11주차| Edge 전처리 방식 , 실시간 검출 안정화, 추론 테스트 | ✅        | 다양한 시스템 구성 시도 |

---

## 🚀 향후 계획

- [ ] 전력측정/성능측정 code 고안 + Inference 결과 반환 형식 정리
- [ ] Edge-Cloud 간 전송 데이터 포맷 최적화 및 경량화 모델 개발
- [ ] 경량화 모델 개발 후 결과 분석 및 시각화(정확도, 속도, 전력)
- [ ] 심화목표 탐구 및 부족한 부분 보완, github 정리
- [ ] 전체 시스템 통합 테스트 및 발표자료/최종보고서 작성

---

## 📎 참고

> ⚠️ 본 프로젝트는 현재 진행 중이며, GitHub에 버전 관리된 코드와 실험 결과가 지속적으로 업데이트되고 있습니다.
> 미완성 코드도 commit 및 branch를 통해 관리되고 있습니다.

- 📌 [프로젝트 GitHub 링크](https://github.com/sehoon120/Embeded_Edge_Cloud)
- 📄 얼굴 인식 모델: InsightFace, ArcFace 등
- 📷 카메라 모듈: Raspberry Pi CSI camera + Picamera2

---