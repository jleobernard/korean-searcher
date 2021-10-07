from com.leo.koreanparser.dl.conf import TARGET_HEIGHT, TARGET_WIDTH
from com.leo.koreanparser.dl.model import get_model, ModelLoss
from com.leo.koreanparser.dl.utils.data_utils import parse_args
from com.leo.koreanparser.dl.utils.train_utils import do_load_model, check_worst_samples

args = parse_args()

models_rep = args['models_path']
load_model = 'True' == args['load']
threshold = float(args['threshold'])

model = get_model()
if not do_load_model(models_rep, model):
    print(f"Could not load model")
    exit(-1)


loss = ModelLoss([float(args['alpha']), float(args['beta']), float(args['gamma']), float(args['theta'])],
                 width=TARGET_WIDTH, height=TARGET_HEIGHT)

batch_size = int(args["batch_size"])

check_worst_samples(model, data_dir=args["datadir"], working_dir=args["working_dir"], threshold=threshold, loss_computer=loss)
