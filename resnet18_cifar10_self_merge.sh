for max_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  for threshold in 0.58 0.59 0.60 0.61 0.62 0.63 00.64 0.65 0.66 0.67 0.68 0.69 0.7
  do
    CUDA_VISIBLE_DEVICES=1 python resnet18_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet18/vanilla/sam0.0/seed0/_resnet18_training_vanilla_sam_3.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
    CUDA_VISIBLE_DEVICES=1 python resnet18_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet18/vanilla/sam0.1/seed0/_resnet18_training_vanilla_sam_2.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
  done
done
