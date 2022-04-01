# The name of experiment
        # --test test \

name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python "src/vqa.py" \
        --train train \
        --valid val \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 10 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load /home/ubuntu/VL-T5/snap/pretrain/VLT5/Epoch30 \
        --num_beams 3 \
        --batch_size 80 \
        --valid_batch_size 100 \