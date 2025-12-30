uv run main.py \
    --batch_size 4 \
    --exp_id my_tennis_exp \
    --num_epochs 300 \
    --lr 1e-4 \
    --val_intervals 5 \
    --steps_per_epoch 1000            # Шагов на эпоху (ограничение, если датасет большой)
