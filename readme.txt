Testing instructions:
1. modify configs/config_test_scope_Final.py, replace $testset_root$ with your test date path and $store_path$ with the output path.
2. use the following command to start inference:
CUDA_VISIBLE_DEVICES=0 python interpolate_REDS_VTSR_MS.py configs/config_test_scope_Final.py

The output results will be stored in $store_path$ written in configs/config_test_scope_Final.py



Note: You may need to compile the scopeflow correlation operation first.
1. compile the correlation operation:
cd models/scopeflow_models/correlation_package
python setup.py install

if you use CUDA>=9.0, just execute the above commands straightforward;
if you use CUDA==8.0, you need to change the folder name 'correlation_package_init' into 'correlation_package', and then execute the above commands.




If you have any question, please contact liuyihao14@mails.ucas.ac.cn



