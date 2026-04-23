export PYTHONPATH=$(pwd):$PYTHONPATH

python src/utils/preprocess_training_data.py --data_dir CVPR-BiomedSegFM/3D_train_npz_all --output_dir CVPR-BiomedSegFM/3D_train_npz_all_processed
python src/utils/generate_train_path_json.py --data_dir CVPR-BiomedSegFM/3D_train_npz_all_processed
python src/utils/generate_validation_data.py