
camtype=("basler" "webcam_c")
for cam in "${camtype[@]}"; do
    for i in {1..5}; do
            python train.py -s /data1/ltw/video-head-tracker-main/data/EVE_val0${i}_${cam}_10 -m output/EVE_val0${i}_${cam}_full_v5 --white_background --bind_to_mesh --eye_loss --eye_mask --z_rotate --add_lpips --load_light
    done
done