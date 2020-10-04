testset_root = '/data1/yhliu/REDS_VTSR/val/val_15fps/'   # put your testing input folder here
test_size = (1280, 720)          # speficy the frame resolution
test_crop_size = (1280, 720)

mean = [0.0, 0.0, 0.0]
std  = [1, 1, 1]

inter_frames = 1     # number of interpolated frames
preserve_input = False  # whether to preserve the input frames in the store path


model = 'AcSloMoS_scope_unet_residual_synthesis_edge_LSE'  
pwc_path = './utils/network-default.pytorch'


store_path = 'outputs/release_val_Stage123_scratch_vgg'          # where to store the outputs
checkpoint = 'checkpoints/Stage123_scratch_vgg/Stage123_scratch_vgg_checkpoint.ckpt'

