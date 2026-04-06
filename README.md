EM-MambaNet

![24](https://github.com/user-attachments/assets/0aaacfa6-0352-4632-b006-f350823cf2a1)


🧠 Introduction

An edge-guided enhanced change detection network, termed Edge-injected and Modulation-guided Mamba Network (EM-MambaNet), is proposed to further improve detection accuracy and alleviate the problems of blurred detail changes and missed detections in weak boundary change regions. The network designs an enhancement mechanism that jointly employs an edge injection module and an edge modulation module. The former explicitly introduces edge structural priors into the shallow layers of the encoder to strengthen the perception of object contours. The latter utilizes stable edge information extracted from SAR features to perform adaptive spatial enhancement on the optical branch, thereby enabling precise localization and accurate segmentation of change regions.

🌟 Datasets

California、Gloucester I/II、Shuguang from https://github.com/lixinghua5540/MTCDN?tab=readme-ov-fileCD

📌 Training and evaluation

单GPU训练：CUDA_VISIBLE_DEVICES="0" python train.py -d 0 -n "dataset_name"

多GPU训练：NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2  --master_port 29502 train.py -p 29502 -d 0,1 -n "dataset_name
