input_dir=$1

for file in $(ls $input_dir)
do
    ./experiment_file1d.sh $input_dir$file > experiment_$file.csv &
done