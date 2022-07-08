opt=tp_denoisenet_v2_002
epoch=28
postfix='001'

python realworld.py --opt options/${opt}.yaml \
    --num_gpus 1 \
    --model_path snapshot/${opt}/GNet/GNet-epoch-${epoch}.pkl \
    --src_path /mnt/lustre/share/zhaoyuzhi/mobile_phone/long8_short1_20200701 \
    --save_path result/${opt}/epoch-${epoch}/long8_short1_20200701 \
    --enable_patch True \
    --patch_size 320 \
    --save_deblur True \
    --postfix ${postfix}
