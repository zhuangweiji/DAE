#export train_cmd="run.pl --mem 8G"
#export decode_cmd="run.pl --mem 4G"
#export mkgraph_cmd="run.pl --mem 8G"

export train_cmd="queue.pl -q all.q,cpu.q --mem 1G"
export decode_cmd="queue.pl -q all.q,cpu.q --mem 1G"
export mkgraph_cmd="queue.pl -q all.q,cpu.q --mem 1G"
export egs_cmd="queue.pl -q all.q,cpu.q --mem 1G"
export cuda_cmd="queue.pl -q v100.q"
