testset_root = '/home/yhliu/DATA/old_films_afterSR/'   # put your testing input folder here
test_size = (2560, 1440)          # speficy the frame resolution (640, 360) 
test_crop_size = (2560, 1440)

mean = [0.0, 0.0, 0.0]
std  = [1, 1, 1]

inter_frames = 3     # number of interpolated frames
preserve_input = True  # whether to preserve the input frames in the store path


model = 'AcSloMoS_scope_unet_residual_synthesis_edge_LSE'  
pwc_path = './utils/network-default.pytorch'


store_path = 'outputs/old_films_interp3'          # where to store the outputs
checkpoint = 'checkpoints/Stage123_scratch/Stage123_scratch_checkpoint.ckpt'

