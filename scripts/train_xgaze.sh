for i in {48..59}; do
        if ((i<10)); then
            python train.py -s /data1/ltw/video-head-tracker-main/data/xgaze_00${i} -m output/xgaze_00${i}   --eval --white_background --bind_to_mesh --eye_mask --eye_loss --add_lpips --z_rotate
        else
            python train.py -s /data1/ltw/video-head-tracker-main/data/xgaze_0${i} -m output/xgaze_00${i}   --eval --white_background --bind_to_mesh --eye_mask --eye_loss --add_lpips --z_rotate
        fi
    done 