# OSS_Project
파이썬 flask 웹 프레임워크를 이용하여 학습한 모델로 칵테일을 추천하는 웹서비스


## 프로젝트 소개
- 프로젝트 이름 : 칵테일 추천 시스템
- 프로젝트 설명 : 기본적으로는 사용자가 입력한 텍스트를 추출하여 얻은 텍스트로 gpt에게 추천하는 칵테일을 받고 직접 학습한 모델에게 그와 비슷한 칵테일을 추천받는 웹 서비스
- 프로젝트 의의 : 해당 프로젝트는 오픈소스 프로젝트로서 개발하였고 다른 사용자들도 해당 서비스를 이용, 개발 할 수 있도록 합니다. 


### 개발 팀 
- 김진우 : 팀장, db 서비스, 기본적인 유저 기능 개발
- 문병준 : 프론트엔드와 gpt 서비스 개발
- 이민우 : 칵테일 추천 모델 개발
- 인태영 : gpt 서비스 개발
- 허소중 : 술 추천 시스템 BERT 모델 개발


### 개발 기간
- 2024.11.01 ~ 2024.12.12


### 개발 환경
- Python 3.12.5
- WebFramework : Flask 3.1.0
- DB : PostgreSQL 16.4
- Communication : Rocketchat, Redmine


### 기능


##### 칵테일 추천
- 사용자가 입력한 값 추출, 모델로 정제 후 기존의 칵테일 데이터셋에서
  추천하는 칵테일을 gpt에게 칵테일을 받고 그 해당하는 칵테일들의 재료를 다시 칵테일추천모델에게 추천받아서 최종적으로 gpt가 추천하는 칵테일(1 - 3개), 칵테일추천모델이 추천하는 칵테일(1 - 5)로 끝납니다.

- 1. 입력란에 원하는 칵테일을 입력한다.

![4-추천입력](https://github.com/user-attachments/assets/126716b5-1112-44fb-98ef-5232938c2a19)



- 2. 추천받은 칵테일들을 확인한다.

![5_1-추천값출력](https://github.com/user-attachments/assets/c2a53903-e67c-484e-8b5d-ac04c5799110)



- 3. 미완성 칵테일 모델 추천(칵테일데이터셋으로부터 정보를 불러오지 않음)

![6-추천모델값-로그](https://github.com/user-attachments/assets/babd7aad-cbd8-47b4-9cef-98785004435d)


### 아쉬운 점
- 기존에 값들을 제대로 정리하여 html에 출력하고 싶었지만 아직 flask에 미숙했던지라 시간이 부족하였다.
- 특히 사용자가 이전에 입력하였던 추천 칵테일을 db에 저장하여 입력을 하지 않아도 이전에 현재 학습한 모델로 추천받아 칵테일을 추천하는 기능 또한 만들려하였다.
- 후에 flask기반으로하여 개발한다면 기존에 학습했던걸 바탕으로 좀 더 완성된 서비스를 개발하고 싶다.




