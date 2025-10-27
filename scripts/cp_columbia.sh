noglass=(2 3 4 5 6 7 9 13 14 15 18 19 20 21 24 27 29 30 31 33 35 38 40 42 44 45 47 48 49 51 52 53 54 56)

for i in "${noglass[@]}"; do
    echo $i
    if ((i<10)); then
        rm -rf  dataset/columbia/columbia_0${i}/frames/*/iris*
    else
        rm -rf dataset/columbia/columbia_${i}/frames/*/iris*
    fi
    done 