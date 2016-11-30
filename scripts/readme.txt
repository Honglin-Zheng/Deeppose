python scripts/train_vgg16.py \
--epoch 2 \
--batchsize 128 \
--lr 0.05 \
--snapshot 10 \
--datadir data/FLIC-full \
--channel 3 \
--test_freq 10 \
--flip 1 \
--size 224 \
--min_dim 100 \
--cropping 1 \
--crop_pad_inf 1.4 \
--crop_pad_sup 1.6 \
--shift 5 \
--gcn 1 \
--joint_num 7 \
--symmetric_joints "[[2, 4], [1, 5], [0, 6]]" \
--opt Adam \
--weights_path "vgg16_weights.h5"  >> out.txt

Training procedure:
python scripts/train_vgg16.py --epoch 2 --batchsize 64 --times_per_batch 1 --datadir data/FLIC-full --partial_dataset 1000 --channel 3 --size 224 --min_dim 100 --cropping 1 --crop_pad_inf 1.4 --crop_pad_sup 1.6 --shift 5 --joints_num 7 --opt Adam --weights_path "vgg16_weights.h5" --untrainable_layers 32 --debug 1

Testing procedure
python scripts/test_vgg16.py --opt Adam --batchsize 64 --datadir data/FLIC-full --modeldir results/vgg16_2016-08-16_23-26-31 --partial_dataset 0 --channel 3 --size 224 --min_dim 100 --cropping 1 --crop_pad_inf 1.4 --crop_pad_sup 1.6 --shift 5 --joints_num 7 --draw_image 0

Plot joints procedure
python scripts/test_vgg16.py --datadir data/FLIC-full --predictdir . --channel 3 --size 224 --min_dim 100 --cropping 1 --crop_pad_inf 1.4 --crop_pad_sup 1.6 --shift 5 --draw_image 1
