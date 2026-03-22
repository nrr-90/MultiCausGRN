from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import MultiCausGRN
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
Non-Specific
mHSC-L learning rate = 3e-5
"""
data_type = 'PBMC_MULTIOME_A'
num = 12786


def embed2file(tf_embed, tg_embed, gene_names, tf_path, target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    tf_df = pd.DataFrame(tf_embed, index=gene_names)
    tg_df = pd.DataFrame(tg_embed, index=gene_names)

    tf_df.to_csv(tf_path)
    tg_df.to_csv(target_path)




#density = Network_Statistic(data_type,num,net_type)
exp_file = '/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/ExpressionData_clean.csv'
tf_file = "/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/TF.csv"
target_file = "/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/Target.csv"

data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file)["index"].astype("int64").values
target = pd.read_csv(target_file)["index"].astype("int64").values
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

train_file = "/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/Train_set2.csv"
test_file = "/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/Validation_set2.csv"
val_file = "/content/drive/MyDrive/MultiCausGRN/PBMC_multiome_preprocessed/PBMC/Model_INPUT_A/Test_set2.csv"


tf_embed_path = r'Result/MultiOmics2/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/MultiOmics2/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/MultiOmics2/'+data_type+' '+str(num)):
    os.makedirs('Result/MultiOmics2/'+data_type+' '+str(num))

train_data = pd.read_csv(train_file).values
validation_data = pd.read_csv(val_file).values
test_data = pd.read_csv(test_file).values

# safety cast (even after rewrite)
train_data[:,0] = train_data[:,0].astype(np.int64)
train_data[:,1] = train_data[:,1].astype(np.int64)
train_data[:,2] = train_data[:,2].astype(np.float32)

validation_data[:,0] = validation_data[:,0].astype(np.int64)
validation_data[:,1] = validation_data[:,1].astype(np.int64)
validation_data[:,2] = validation_data[:,2].astype(np.float32)

test_data[:,0] = test_data[:,0].astype(np.int64)
test_data[:,1] = test_data[:,1].astype(np.int64)
test_data[:,2] = test_data[:,2].astype(np.float32)

print("train row:", train_data[0], train_data[:,0].dtype, train_data[:,1].dtype)
print("train pos rate:", train_data[:,-1].mean())
print("val pos rate:", validation_data[:,-1].mean())
print("test pos rate:", test_data[:,-1].mean())
train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf,loop=args.loop)


adj = adj2saprse_tensor(adj)


train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

model = MultiCausGRN(input_dim=feature.size()[1],
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                )


adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)


optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)



for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)


        # train_y = train_y.to(device).view(-1, 1)
        pred = model(data_feature, adj, train_x)

        #pred = torch.sigmoid(pred)
        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)


        loss_BCE.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_BCE.item()


    model.eval()
    score = model(data_feature, adj, validation_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    # score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
        #
    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))

torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')

model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
model.eval()
tf_embed, target_embed = model.get_embedding()
gene_names = data_input.index.astype(str).tolist()
embed2file(tf_embed, target_embed, gene_names, tf_embed_path, target_embed_path)
score = model(data_feature, adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)
# score = torch.sigmoid(score)


AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)

print(f"FINAL_RESULT seed={args.seed} "
      f"val_AUPR={best_val_aupr:.6f} "
      f"test_AUC={AUC:.6f} "
      f"test_AUPRC={AUPR:.6f}")
import csv
file_exists = os.path.isfile("results.csv")

with open("results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", "val_AUPR", "test_AUC", "test_AUPRC"])
    writer.writerow([args.seed, best_val_aupr, AUC, AUPR])





















