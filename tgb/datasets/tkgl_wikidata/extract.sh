for chunk in 5; do
num_chunk=25
while [ $chunk -le $num_chunk ]; do
    cmd="tkgl_wikidata.py \
    --chunk ${chunk} \
    --num_chunks ${num_chunk} \
    "
    python $cmd
    chunk=$(( $chunk + 1 ))
done
done