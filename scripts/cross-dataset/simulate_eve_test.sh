camtype=("webcam_l" "webcam_r")
for cam in "${camtype[@]}"; do
    for i in {1..10}; do
        if ((i<10)); then
            python simulate.py --model_path output/EVE_test0${i}_${cam}_mp --dataset_type eve --data_size 750 --normalize_eye --output_h5
            # python vis_paper.py --model_path output/EVE_test0${i}_${cam}

        else
            python simulate.py --model_path output/EVE_test${i}_${cam}_mp --dataset_type eve --data_size 750 --normalize_eye --output_h5
            # python vis_paper.py --model_path output/EVE_test${i}_${cam}
        fi
    done
done