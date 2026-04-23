export PYTHONPATH=$(pwd):$PYTHONPATH

deepspeed src/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_dir CVPR-BiomedSegFM/3D_train_npz_all_processed \
    --text_model medicalai/ClinicalBERT \
    --image_size 96 96 96 \
    --patch_size 16 16 16 \
    --pass_num 1 \
    --transformer_depth 6 \
    --mlp_dim 3072 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 12 \
    --output_dir output/ESICA \
    --run_name ESICA

deepspeed src/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_dir CVPR-BiomedSegFM/3D_train_npz_all_processed_v3 \
    --image_size 96 96 96 \
    --pretrained_model output/ESICA \
    --pos_num 1 \
    --neg_num 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-6 \
    --freeze_text_encoder True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 12 \
    --output_dir output/ESICA_Finetune \
    --run_name ESICA_Finetune