record_dir = 'Reproduce001_EQVI_from_scratch_5Lap_10L1'
checkpoint_dir = 'checkpoints/Reproduce001_EQVI_from_scratch_5Lap_10L1'
trainset = 'VTSR'
trainset_root = '/home/yhliu/DATA/train_60fps/'
train_size = (1280, 720)
train_crop_size = (512, 512)

validationset = 'REDS_val'
validationset_root = '/data1/yhliu/REDS_VTSR/select_val_60fps/'
validation_size = (1280, 720)
validation_crop_size = (1280, 720)


train_batch_size = 12


train_continue = False
epochs = 200
progress_iter = 439

min_save_epoch = 30
checkpoint_epoch = 1


mean = [0.0, 0.0, 0.0]
std  = [1, 1, 1]



model = 'AcSloMoS_scope_unet_residual_synthesis_edge_LSE'  # AcSloMoS_lsr_scope | AcSloMoS_scope_unet_residual_synthesis_edge_LSE
pwc_path = './network-default.pytorch'

init_learning_rate = 1e-4
milestones = [100, 150]
