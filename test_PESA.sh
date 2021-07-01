clear all
python main_all.py --run_mode=test --model_path=./log_PESA/train/ --res_path=./log_PESA/test/ --gpu_id='0'--split=1 --group=4 --model='PESA_Net' --use_ransac=True --data_te=./dataset/yfcc-sift-2000-test.hdf5 --train_lr=1e-3

