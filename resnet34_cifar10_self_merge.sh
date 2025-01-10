for max_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  for threshold in 0.60 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80
  do
    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.0/seed0/_resnet34_training_vanilla.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.0/seed1/_resnet34_training_vanilla.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.0/seed2/_resnet34_training_vanilla.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold

    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.1/seed0/_resnet34_training_vanilla_noisy_1.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.1/seed1/_resnet34_training_vanilla_noisy_1.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold
    CUDA_VISIBLE_DEVICES=1 python resnet34_cifar10_self_weight_matching.py --model_param /cpfs01/user/chenzijun/zhouzhanpeng/Iterative-Feature-Merging/output/train/cifar10/resnet34/vanilla/noise0.1/seed2/_resnet34_training_vanilla_noisy_1.yml/params/model_epoch159.pt --max_ratio $max_ratio --threshold $threshold

  done
done
