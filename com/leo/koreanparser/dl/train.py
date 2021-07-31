import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from com.leo.koreanparser.dl.model import get_model
from com.leo.koreanparser.dl.utils.data_utils import load_train_data, parse_args, SubsDataset
from com.leo.koreanparser.dl.utils.train_utils import train_epocs

args = parse_args()

df_train = load_train_data(args["datadir"])
df_train = df_train.reset_index()
X = df_train[['new_path', 'new_bb']]
Y = df_train['subs']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
train_ds = SubsDataset(X_train['new_path'], X_train['new_bb'], y_train, transforms=True)
valid_ds = SubsDataset(X_val['new_path'], X_val['new_bb'], y_val)

batch_size = int(args["batch_size"])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, drop_last=True)

model = get_model()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.006)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=int(args['epochs']))
