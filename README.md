# Review-Based Personalized Restaurant Recommendation System

서울 음식점 데이터를 수집하고, 리뷰 텍스트와 사용자 평점 정보를 결합해 **개인 맞춤형 음식점 추천**을 수행한 하이브리드 추천 시스템 프로젝트입니다.

---

## Overview

기존 음식점 추천은 인기순, 평점순 중심이라 사용자 취향을 충분히 반영하기 어렵습니다.  
이 프로젝트는 음식점 리뷰와 사용자 평점 데이터를 활용해 **리뷰 기반 취향 정보 + 사용자-아이템 상호작용 정보**를 함께 반영하는 추천 시스템을 구현한 프로젝트입니다.

---

## Tech Stack

- **Language**: Python
- **Data**: Pandas, NumPy
- **Crawling**: Selenium, BeautifulSoup
- **NLP / Features**: NLTK, googletrans, Gensim
- **Modeling**: scikit-learn, Cornac, NeuMF
- **Visualization**: Matplotlib, Folium

---

## Pipeline

```text
Data Crawling
→ Data Cleaning & Preprocessing
→ Review Translation
→ Tag Extraction
→ Word2Vec Embedding
→ KMeans Clustering
→ User / Restaurant Matrix Construction
→ CBF / NCF Modeling
→ Hybrid Recommendation
→ Evaluation
```

---

## Key Features

- 다이닝코드 기반 음식점/리뷰 데이터 크롤링
- 리뷰 전처리 및 번역
- 형용사 중심 태그 추출
- Word2Vec + KMeans 기반 특징 추출
- CBF + NCF 기반 하이브리드 추천
- NDCG@10 기반 성능 평가
- 지도 기반 추천 결과 시각화

---

## Repository Structure

```text
restaurant-recommendation-system/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ images/
├─ src/
│  ├─ main.py
│  ├─ crawling/
│  ├─ preprocessing/
│  ├─ features/
│  ├─ recommend/
│  └─ evaluation/
└─ outputs/
```

---

## Models

### 1. Content-Based Filtering (CBF)
사용자 리뷰 성향과 음식점 리뷰 특징의 유사도를 기반으로 추천합니다.

### 2. Neural Collaborative Filtering (NCF)
사용자-음식점 평점 데이터를 기반으로 잠재 선호를 학습합니다.

### 3. Hybrid Recommendation
CBF와 NCF 점수를 결합해 최종 추천 결과를 생성합니다.

---

## Evaluation

주요 평가지표는 **NDCG@10**이며, NeuMF 실험 결과는 다음과 같습니다.

- **NDCG@10 = 0.8401**

<img width="413" height="163" alt="image" src="https://github.com/user-attachments/assets/20a0f9ed-49a6-4007-8e12-d4f8980f754f" />


---

## How to Run

### 1) Install

```bash
git clone https://github.com/your-github-id/restaurant-recommendation-system.git
cd restaurant-recommendation-system
pip install -r requirements.txt
```

### 2) Run

```bash
python -m src.main crawl
python -m src.main preprocess
python -m src.main features
python -m src.main recommend
```

전체 실행:

```bash
python -m src.main all
```

---

## Limitations

- 크롤링 대상 사이트 구조 변경 시 selector 수정이 필요합니다.
- 한국어 리뷰를 영어로 번역한 뒤 임베딩을 수행해 정보 손실 가능성이 있습니다.
- 베이스라인 비교와 평가 설계는 추가 개선 여지가 있습니다.

---

## Contribution

- 음식점/리뷰 데이터 크롤링
- 리뷰 전처리 및 데이터 구조화
- 추천 모델링 파이프라인 구성
- Geocoding 기반 시각화
- 결과 분석 및 프로젝트 정리
