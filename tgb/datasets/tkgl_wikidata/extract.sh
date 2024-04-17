for chunk in 2; do
num_chunk=55 
while [ $chunk -le $num_chunk ]; do
    cmd="tkgl_wikidata.py \
    --chunk ${chunk} \
    --num_chunks ${num_chunk} \
    "
    python $cmd
    chunk=$(( $chunk + 1 ))
done
done