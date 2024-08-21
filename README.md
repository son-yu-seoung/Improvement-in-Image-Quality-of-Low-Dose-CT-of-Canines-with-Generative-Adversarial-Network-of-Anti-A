# Improvement in Image Quality of Low-dose CT of Canines with Generative Adversarial Network of Anti-aliasing Generator and Multi-scale Discriminator


우리는 unpaired 저선량과 표준 선량 CT images를 Input과 Output으로 해 Unsupervised Learning 방식으로 학습하는 Framework를 제안한다.
아래는 우리의 연구 paper에 접근할 수 있는 링크이다. 

[[paper]](naver.com)

---
# Getting Started
- Clone this repo:

```
git clone https://github.com/son-yu-seoung/Low-dose-CT-Denoising-Framework.git
cd Low-dose-CT-Denoising-Framework
```

- Install ours requirments:

```
pip install -r requirements.txt 
```

- Train
```
python ImgPreparation_CycleGAN.py 
python CycleGAN_train.py 
```

- Test

```
python CycleGAN_predict.py  
```
