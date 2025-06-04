image_root=r'./datasets/rec_data/'
split_root=r'./datasets/rec_data/data/'
 
output_dir=r'outputs/DIOR_R101_P800'
batch_size=20
lr_drop=60
epochs =100

dataset='DIOR_RSVG'  # referit, unc, unc+, gref, gref_umd, flickr, DIOR_RSVG, OPT_RSVG, VRSBench_Ref
imsize=800
max_query_len=40
model_type='ResNet'  # 
detr_model='checkpoints/detr-r101-2c7b67e5.pth'

backbone='resnet101'  # resnet101
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
vl_dec_num=4  # 采样解码数量
uniform_learnable=True 
uniform_grid=False
in_points=32

# 下面是消融实验的设置
# 视觉和语言特征从骨干网络中提取后，在解码前的交互部分的编码层控制
vl_enc_num=0  # vltrans的层数目