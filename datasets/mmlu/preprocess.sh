cd ../..

python tools/preprocess_data.py \
       --input datasets/mmlu/mmlu_train.json \
       --output-prefix mmlu \
       --vocab-file datasets/gpt2-vocab.json \
       --merge-file datasets/gpt2-merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --json-key "question" \
       --workers 5

mv mmlu* datasets/mmlu
