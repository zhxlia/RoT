SPLIT=test
for model in Llama3.1;
do
    for scale in 8B;
    do
        for DATASET in WikiTQ;
        do
            for SPLIT in test;
            do
                DATA_FILE="./dataset/${DATASET}/$SPLIT.list.json"
                if [ ! -f "$DATA_FILE" ]; then
                    echo "$DATA_FILE does not exist, continuing to next dataset."
                    continue
                fi

                DUMP_FILE=./inference/results/$DATASET/$model/$scale/inference_traverse
                [ ! -d "$DUMP_FILE" ] && mkdir -p "$DUMP_FILE"

                python3 ./inference/inference_traverse.py \
                        --model ./model/$model/$scale \
                        --config_file ./config/Llama3.1.json \
                        --questions_file $DATA_FILE \
                        --shot_num 1 \
                        --dump_file $DUMP_FILE/$SPLIT.s{shot_num}.json \
                        # --data_size 64
            done
        done
    done
done
