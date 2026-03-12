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
