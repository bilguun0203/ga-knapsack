script=$1
input_dir=$2
iterations=$3
generations='100 750 1500'
selections='roulette'
populations='50 100 150'
mutation_probs='0.001 0.01'
echo "FILE, GENERATION, SELECTION, POPULATION, MUTATION_PROB, SOLUTION, OPTIMAL DIFF, DURATION (MS)"

for file in $(ls $input_dir)
do
    for generation in $generations
    do
        for selection in $selections
        do
            for population in $populations
            do
                for mutation_prob in $mutation_probs
                do
                    for ((i=0; i<iterations; i++));
                    do
                        STARTTIME=$(date +%s%N)
                        solution=$(python $script $input_dir$file -g $generation -p $population -s $selection -m $mutation_prob --silent --optimal)
                        ENDTIME=$(date +%s%N)
                        echo "$file, $generation, $selection, $population, $mutation_prob, $solution, $((($ENDTIME - $STARTTIME) / 1000000))"
                    done
                done
            done
        done
    done
done
