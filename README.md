# Improvement in Image Quality of Low-dose CT of Canines with Generative Adversarial Network of Anti-aliasing Generator and Multi-scale Discriminator


### Abstract [[paper]](https://www.mdpi.com/2306-5354/11/9/944)
Computed tomography (CT) imaging is vital for diagnosing and monitoring diseases in both humans and animals, yet radiation exposure remains a significant concern, especially in animal imaging. Low-dose CT (LDCT) minimizes radiation exposure but often compromises image quality due to a reduced signal-to-noise ratio (SNR). Recent advancements in deep learning, particularly with CycleGAN, offer promising solutions for denoising LDCT images, though challenges in preserving anatomical detail and image sharpness persist. This study introduces a novel framework tailored for animal LDCT imaging, integrating deep learning techniques within the CycleGAN architecture. Key components include BlurPool for mitigating high-resolution image distortion, Pixelshuffle for enhancing expressiveness, Hierarchical Feature Synthesis (HFS) networks for feature retention, and spatial-channel squeeze-excitation (scSE) blocks for contrast reproduction. Additionally, a multi-scale discriminator enhances detail assessment, supporting effective adversarial learning. Rigorous experimentation on veterinary CT images demonstrates our framework's superiority over traditional denoising methods, achieving significant improvements in noise reduction, contrast enhancement, and anatomical structure preservation. Extensive evaluations using precision and recall metrics validate our approach's efficacy, highlighting its potential to enhance diagnostic accuracy in veterinary imaging. Ablation studies confirm the scSE method's critical role in optimizing performance, and robustness to input variations underscores its practical utility.


<p align='center'>
  <img src="https://github.com/user-attachments/assets/175793d6-ccf4-4e5b-9324-318d7abc21e4"> 
</p>
<div align=center> 
Figure 1. Low-dose CT Denoising Framework, X is LDCT, Y is SDCT
</div>


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
