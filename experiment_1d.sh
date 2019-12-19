input_dir='./inputs/1d/'
generations='100 500 1000 1500'
selections='group roulette'
populations='20 40 60'
mutation_probs='0.001 0.01 0.5'
echo "FILE, GENERATION, SELECTION, POPULATION, MUTATION_PROB, SOLUTION, DURATION (MS)"
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
                    for i in {1..20}
                    do
                        STARTTIME=$(date +%s%N)
                        solution=$(python GAKnapsack.py $input_dir$file -g $generation -p $population -s $selection -m $mutation_prob --silent)
                        ENDTIME=$(date +%s%N)
                        echo "$file, $generation, $selection, $population, $mutation_prob, $solution, $((($ENDTIME - $STARTTIME) / 1000000))"
                    done
                done
            done
        done
    done
done
echo 'Done.'