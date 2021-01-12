python -m apps.train_shape --dataroot C:\Blueprint2Car\data ^
 --gpu_ids 0 ^
 --skip_downsample ^
 --batch_size 2 ^
 --use_normal_loss ^
 --mlp_type conv1d ^
 --num_sample_inout 5000 ^
 --num_sample_normals 5000 ^
 --skip_downsample ^
 --hourglass_dim_internal 64^
 --hourglass_dim 64 ^
 --resolution 256 ^
 --loadSize 256 ^
 --num_views 4 ^
 --num_threads 6

