
camtype=("basler" "webcam_c" "webcam_l" "webcam_r")
for cam in "${camtype[@]}"; do
    for i in {2..2}; do
        if ((i<10)); then
            python simulate.py --model_path output/EVE_test0${i}_${cam}_mp --dataset_type eve --data_size 10 --anno 
            # python vis_paper.py --model_path output/EVE_test0${i}_${cam}

        else
            python simulate.py --model_path output/EVE_test${i}_${cam}_mp --dataset_type eve --data_size 10 --anno
            # python vis_paper.py --model_path output/EVE_test${i}_${cam}
        fi
    done
done