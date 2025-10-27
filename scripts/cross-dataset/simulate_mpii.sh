for i in {0..14}; do
            if ((i<10)); then
                python simulate.py --model_path output/mpii_0${i}_illum_full_v5  --dataset_type mpii --data_size 4000 --load_light --output_h5
            else
                python simulate.py --model_path output/mpii_${i}_illum_full_v5  --dataset_type mpii --data_size 4000  --load_light --output_h5
            fi
    done
