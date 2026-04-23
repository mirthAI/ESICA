export PYTHONPATH=$(pwd):$PYTHONPATH

python src/eval/pred.py --model_path output/ESICA_Finetune --input_dir CVPR-BiomedSegFM/3D_val_npz --output_dir output/eval/ESICA_Finetune
python src/eval/eval.py --input_dir output/eval/ESICA_Finetune