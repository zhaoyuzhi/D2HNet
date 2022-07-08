opt=tp_deblurnet_v4_002
epoch=30
postfix='_long_last'
valres=1440

python validation.py --opt options/${opt}.yaml \
    --num_gpus 1 \
    --model_path snapshot/${opt}/GNet/GNet-epoch-${epoch}.pkl \
    --val_path '/mnt/lustre/share/zhaoyuzhi/slrgb2rgb_simulated_mobile_phone/val_no_overlap_noisy_' \
    --val_res ${valres} \
    --val_sharp_path '/mnt/lustre/share/zhaoyuzhi/slrgb2rgb_simulated_mobile_phone_sharp/val_no_overlap' \
    --save_path result-val-${valres}/${opt} \
    --down_img_size 1024 \
    --enable_patch False \
    --patch_size 1024 \
    --save_deblur False \
    --save_residual False \
    --postfix ${postfix}
