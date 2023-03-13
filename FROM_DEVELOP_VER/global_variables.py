
nsteps=199
num_variables = 4 # .... flowfield variables
num_pred      = 4 # .... input q
num_dynamics  = 4
num_geoInfos  = 11
num_label_para = 0
idim = 129
jdim = 129

imax=idim 
jmax=jdim 



wake_lower_jst=1
wake_lower_jfn=41
wake_upper_jst=217
wake_upper_jfn=257





jst=41
jfn=217
nwalls = 2
n_sampling = 10


vectorSize = num_variables*(jfn-jst+1)*nwalls


# number of training iterations
iterations = 200000
#iterations =   8000
# batch size
batch_size = 1
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo =4 #7 --overfitting
# data set config
prop=None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
dropout    = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoadG_1     = "modelG_1"      # optional, path to pre-trained model
doLoadG_2     = "modelG_2"      # optional, path to pre-trained model
#seed = 3483678993

saveL1 = True
saveModel = True
n_save_model = 100
DEBUG_SEQ = False

shuffle_1D = False
##########################

WITH_LABEL_PARA = False
WITH_METRICS = False
WITH_MESH = False
allDirList = [
#"../50c/laminar_0deg_m0p8/data_raw/" # no problem
#,
#"../Pitch0012_aerosurf_run1/train_p3d/",
"../../Ma0.1_movies/train_p3d/", 
#"../Ma0.2_movies/train_p3d/", 
#"../Ma0.3_movies/train_p3d/", 
#"../Ma0.364_movies/train_p3d/",
#"../Pitch0012_aerosurf_run2/train_p3d/",
#"../Pitch0012_aerosurf_run3/train_p3d/",
#"../Pitch0012_aerosurf_run4/train_p3d/",
#"../Pitch0012_aerosurf_run5/train_p3d/"
#,
#"../50c/laminar_0deg_m0p85/data_raw/" # no problem
#"../50c/1_dir/",
#"../50c/2_dir/"
]

labelList = [
#[0.1],
#[0.2],
#[0.3],
]
