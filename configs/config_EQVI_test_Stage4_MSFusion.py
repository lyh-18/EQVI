testset_root = '/data1/yhliu/REDS_VTSR/val/val_15fps/'   # put your testing input folder here
test_size = (1280, 720)          # speficy the frame resolution
test_crop_size = (1280, 720)

mean = [0.0, 0.0, 0.0]
std  = [1, 1, 1]

inter_frames = 1     # number of interpolated frames
preserve_input = False  # whether to preserve the input frames in the store path

model = 'MS_Model_Fusion'   # AcSloMoS_scope_unet_residual_synthesis_LSE | MS_Model_Fusion | AcSloMoS_scope_unet_residual_synthesis_edge_LSE
pwc_path = './utils/network-default.pytorch'


store_path = 'outputs/release_Stage4_MSFusion'
checkpoint = 'checkpoints/Stage4_MSFuion/Stage4_checkpoint.ckpt'

