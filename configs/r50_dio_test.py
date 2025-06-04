image_root=r'./datasets/rec_data/'
split_root=r'./datasets/rec_data/data/'

# test
eval_set='test' # 'DIOR, OPT: val, test'
eval_model=r'/best_checkpoint.pth'
output_dir=r'outputs/eval'
batch_size=20

dataset='DIOR_RSVG'  # referit, unc, unc+, gref, gref_umd, flickr, DIOR_RSVG, OPT_RSVG, VRSBench_Ref
imsize=800
max_query_len=40
model_type='ResNet'  # 
detr_model=None

backbone='resnet50'  # resnet101
bert_enc_num=12
detr_enc_num=6

# dropout
dropout=0.1
vl_dropout=0.1

# enc
vl_enhancevit=True
vl_crosAttn=True
# dynamic dec
st_dec_dyn=True 
vl_dec_num=4  # the number of decoding layers
uniform_learnable=True 
uniform_grid=False
in_points=32

# Below is the setup of the ablation experiment
# After the visual and textual features are extracted from the backbone network, the encoding layer controls the interaction before decoding.
vl_enc_num=0  