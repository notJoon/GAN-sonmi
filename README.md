# GAN-sonmi
 적대적 생성 모델(GAN)을 이용한 이미지 생성

## 설명 
기본적이고 단순한 GAN 모델을 통해서 데이터셋과 푀대한 비슷한 이미지를 생성해냅니다.
특히, 그림을 그리는 작가의 특성을 정확하게 뽑아내는것이 목표입니다. 

---

## 환경 
* Python >= 3.9
* pytorch >= 1.10.0 

---

## 코드 파일
### models 
* `Generator`, `Discriminator`: 기본적인 [DCGAN](https://arxiv.org/abs/1511.06434)의 구현

* `Conditional`: [CGAN](https://arxiv.org/abs/1411.1784) 구현체 

* `Critic`: 와서스테인GAN([WGAN](https://arxiv.org/abs/1701.07875))의 Discriminator. 생성모델은 DCGAN의 Generator를 사용합니다. 

### functions
GAN 모델의 계산을 도와주는 함수들이 있습니다. 