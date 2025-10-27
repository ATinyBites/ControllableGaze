for i in {0..0}; do
        if ((i<10)); then
            python simulate.py  --model_path output/xgaze_00${i}   --dataset_type xgaze --data_size 100  --white_background
        else
            python simulate.py --model_path output/xgaze_00${i}   --dataset_type xgaze --data_size 100 --white_background
        fi
    done 