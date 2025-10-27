for i in {0..14}; do
            if ((i<10)); then
                rm -rf /data2/ltw/ControllableGaze/dataset/mpii/mpii_0${i}/invalid_frames
            else
                rm -rf /data2/ltw/ControllableGaze/dataset/mpii/mpii_${i}/invalid_frames
            fi
done