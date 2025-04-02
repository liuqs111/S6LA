
# S6LA

From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics

## Quick Start

### Train with ResNet on ImageNet-1K

To train ResNet with S6LA, batch size of 4x256 on 4 GPUs

  ```bash
  python train.py -a mamba_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 imagenet_path
  ``` 


### Train with DeiT on ImageNet-1K

To train DeiT-T with S6LA, batch size of 4x256 on 4 GPUs

  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 --use_env main1.py --model swintransformer_s6_tiny_patch16_224_v1 --batch-size 256 --data-path data_path
  ``` 


### Train with SwinTransformer on ImageNet-1K

To train DeiT-T with S6LA, batch size of 4x256 on 4 GPUs

  ```bash
  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py --cfg ../Swin-Transformer/configs/swin/swin_base_patch4_window7_224_s6.yaml --data-path data_path
  ``` 

### Train with PVT on ImageNet-1K

To train DeiT-T with S6LA, batch size of 4x256 on 4 GPUs

  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 --use_env main.py --config ../PVT/classification/configs/pvt_v2/pvt_v2_b2.py --data-path data_path
  ``` 



### MMDetection

Training with our ResNet based model for detection task

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py configs/faster_rcnn/faster_rcnn_r50s6eca_fpn_1x_coco.py --cfg-options data.samples_per_gpu=4
  ```
