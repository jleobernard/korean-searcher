from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.utils.data_utils import parse_args
from com.leo.koreanparser.dl.utils.train_utils import do_load_model, predict_all

args = parse_args()

models_rep = args['models_path']
load_model = 'True' == args['load']
threshold = float(args['threshold'])

model = get_model()
if not do_load_model(models_rep, model):
    print(f"Could not load model")
    exit(-1)


predict_all(model, data_dir=args["datadir"], threshold=threshold)
