from types import SimpleNamespace

args = SimpleNamespace()

## Training Configuration
args.lr = 1e-4
args.beta1 = 0.5
args.beta2 = 0.999 

## Test Model
args.image_size = 256       # 256 x 256
args.g_conv_dim = 64
args.d_conv_dim = 64 
args.g_repeat = 6
args.d_repeat = 6 

## Dir
args.save = ... 