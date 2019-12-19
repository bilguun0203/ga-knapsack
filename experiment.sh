input_dir='./input/'
generations='100 250 500 750 1000 1500'
selections='group roulette'
populations='20 40 60'
echo "FILE, GENERATION, SELECTION, POPULATION, SOLUTION, DURATION (MS)"
for file in $(ls $input_dir)
do
    for generation in $generations
    do
        for selection in $selections
        do
            for population in $populations
            do
                for i in {1..20}
                do
                    STARTTIME=$(date +%s%N)
                    solution=$(python KnapsackGA.py $input_dir$file -g $generation -p $population -s $selection --silent)
                    ENDTIME=$(date +%s%N)
                    echo "$file, $generation, $selection, $population, $solution, $((($ENDTIME - $STARTTIME) / 1000000))"
                done
            done
        done
    done
done
echo 'Done.'