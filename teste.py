import random


import numpy as np
import pandas as pd
import fire
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset.data_review_teste import ReviewData_test
from framework import Model
import models
import config
from utils.utils import *

from metrics.ndcg import ndcg_metric
from metrics.novelty import novelty
from metrics.diversity import diversity

def test(**kwargs):
    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    if opt.bert == "zeroshot":
        opt.model = f"{opt.model}_ZeroShot"
    elif opt.bert == "finetunning":
        opt.model = f"{opt.model}_FineTunning"

    opt.pth_path = f"checkpoints/{opt.model}_{opt.dataset}_{opt.emb_opt}.pth"

    # PARALLEL CONFIG
    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    print(f"Model: {opt.model}")
    
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    all_data = ReviewData_test(opt.data_root, mode="All")
    '''
    for idx in range(len(all_data)):
        data, score = all_data[idx]
        print(f"Exemplo {idx + 1}:")
        print("Dados:", data)
        print("Score:", score)
        print("-" * 30)
    '''
    all_data_loader = DataLoader(all_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: all in the all dataset")
    predict_loss, test_mse, test_mae, all_prediction = predict(model, all_data_loader, opt)

    dataset_name = kwargs.get('dataset', 'default_dataset').replace('_data', '')
    model_name = kwargs.get('model', 'default_model')
    output_file = f"{dataset_name}{model_name}.csv"

    save_predictions_to_csv(all_prediction, output_file)

    print(f'Previsões foram escritas em {output_file}')
    '''
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['user_id', 'item_id', 'rating'])
        for prediction in all_prediction:
            csv_writer.writerow([str(prediction)]) 

    print(f'Previsões foram escritas em {output_file}')
    '''

# @PREDICT OUTPUT FUNCTION
def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0

    mse_values = []
    mae_values = []
    rmse_values = []
    ndcg_values = []
    precision_values = []
    recall_values = []
    novelty_values = []
    diversity_values = []

    predictions = []

    # MODEL IN EVALUTATION STAGE
    model.eval()

    # WITHOUT GRADIENT
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            test_data = unpack_input(opt, test_data)

            if opt.transnet:
                output, _ = model(test_data)
                output = output[1]
            else:
                output = model(test_data)
            
            for i in range(len(scores)):
                user_id = test_data[2][i].item()
                item_id = test_data[3][i].item()
                score = output[i].item()
                predictions.append((user_id, item_id, score))

            mse_loss = torch.mean((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.mean(abs(output-scores))
            total_maeloss += mae_loss.item()

            rmse, precision, recall = calculate_metrics(scores, output)
            novel = novelty(scores.cpu().tolist(), output.cpu().tolist())
            diver = diversity(scores.cpu().tolist(), output.cpu().tolist())

            mse_values.append(mse_loss.cpu().item())
            rmse_values.append(rmse.item())
            mae_values.append(mae_loss.cpu().item())
            precision_values.append(precision.item())
            recall_values.append(recall.item())
            novelty_values.append(novel)
            diversity_values.append(diver)

    if opt.ranking_metrics:
        iteractions, scores = next(iter(data_loader))
        
        user_ids = set([x[0] for x in iteractions])
        item_ids = set([x[1] for x in iteractions])

        with torch.no_grad():
            for user in user_ids:
                test_data = unpack_input(opt, zip([user], [0]))
                # test_data = unpack_input(opt, zip([user]*len(item_ids), item_ids))

                # Here, I have the itens that my user consume
                user_itens = test_data[4].cpu().tolist()[0]

                n_id_itens = opt.item_num - 1
                user_itens = [x for x in user_itens if x < n_id_itens]

                test_data = unpack_input(opt, zip([user]*len(user_itens), user_itens))

                if opt.transnet:
                    output, _ = model(test_data)
                    output = output[1]
                    if len(output.shape) == 0:
                        break

                else:
                    output = model(test_data)


                # iids, output = test_data[3].cpu(), output.cpu()
                output = output.cpu()
                # iids = [x.item() for x in iids]
                iids = user_itens
                output = [x.item() for x in output]


                item_x_rating = list(zip(iids, output))
                item_x_rating.sort(key=lambda x: x[1])

                list_wise = [x[0] for x in item_x_rating]

                grownd_truth = [y[1] for y in [x for x in iteractions if user == x[0]]]

                ndcg = ndcg_metric(grownd_truth, list_wise, nranks=4)

                ndcg_values.append(ndcg)
    
    if opt.statistical_test:

        df_error = {
            "mse":mse_values, 
            "mae":mae_values,
            "rmse": rmse_values,
            "precision": precision_values,
            "recall": recall_values,
            "diversity": diversity_values,
            "novelty": novelty_values,
            }
        
        df_error = pd.DataFrame(df_error)
        df_error.to_csv(f"results/{opt.model}_{opt.dataset}_{opt.emb_opt}_results_error.csv", index=False)

        df_rank = {
            "ndcg":ndcg_values, 
            }
        
        df_rank = pd.DataFrame(df_rank)
        df_rank.to_csv(f"results/{opt.model}_{opt.dataset}_{opt.emb_opt}_results_rank.csv", index=False)


    else:
        print(f'''MSE mean: {np.array(mse_values).mean():.2f},
                MAE mean: {np.array(mae_values).mean():.2f}, 
                RMSE mean: {np.array(rmse_values).mean():.2f}, 
                NDCG mean: {np.array(ndcg_values).mean():.5f}, 
                PRECISION mean: {np.array(precision_values).mean():.2f},
                RECALL mean: {np.array(recall_values).mean():.2f},
                NOVELTY mean: {np.array(novelty_values).mean():.2f},
                DIVERSITY mean: {np.array(diversity_values).mean():.2f}'''
                
            )

            

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    
    # RETURN TO TRAIN STAGE
    model.train()

    return total_loss, mse, mae, predictions

def unpack_input(opt, x):

    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)
    
    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids] 

    
    user_doc = opt.user_doc[uids]
    
    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids] 
    item_doc = opt.item_doc[iids]

    if opt.topics:
        user_doc = opt.topic_matrix[uids]

        shift = opt.user_num - 2
        item_doc = opt.topic_matrix[[x + shift for x in iids]] 

    
    if opt.bert == "finetunning" or opt.bert == "zeroshot":
        data = [torch.FloatTensor(user_reviews).cuda(), 
                torch.FloatTensor(item_reviews).cuda(),
                torch.LongTensor(uids).cuda(), 
                torch.LongTensor(iids).cuda(), 
                torch.LongTensor(user_item2id).cuda(),
                torch.LongTensor(item_user2id).cuda(), 
                torch.FloatTensor(user_doc).cuda(), 
                torch.FloatTensor(item_doc).cuda()]
        
    else:
        data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
        data = list(map(lambda x: torch.LongTensor(x).cuda(), data))


    return data

def save_predictions_to_csv(predictions, output_file):
    # Cabeçalho do arquivo CSV
    header = ['user_id', 'item_id', 'ratings']

    # Abrir o arquivo CSV para escrita
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Escrever o cabeçalho no arquivo CSV
        csv_writer.writerow(header)
        
        # Escrever as previsões no arquivo CSV
        for prediction in predictions:
            csv_writer.writerow(prediction)


if __name__ == "__main__":
    fire.Fire()
