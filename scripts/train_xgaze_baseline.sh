for i in {3..3}; do
        if ((i<10)); then
            python train.py -s /data1/ltw/video-head-tracker-main/data/xgaze_00${i} -m output/xgaze_00${i}_baseline   --eval --white_background --bind_to_mesh 
        else
            python train.py -s /data1/ltw/video-head-tracker-main/data/xgaze_0${i} -m output/xgaze_00${i}_baseline   --eval --white_background --bind_to_mesh
        fi
    done 