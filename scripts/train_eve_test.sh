camtype=("webcam_r")
for cam in "${camtype[@]}"; do
    for i in {7..10}; do    
        if ((i<10)); then
            python train.py -s /data1/ltw/video-head-tracker-main/data/EVE_test0${i}_${cam}_mp -m output/EVE_test0${i}_${cam}_mp --eval --white_background --bind_to_mesh
        else
            python train.py -s /data1/ltw/video-head-tracker-main/data/EVE_test${i}_${cam}_mp -m output/EVE_test${i}_${cam}_mp --eval --white_background --bind_to_mesh
        fi

    done
done