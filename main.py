# -*- encoding: utf-8 -*-
import time
import random
import math
import fire
import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset import ReviewData
from framework import Model
import models
import config
from utils.utils import *

from metrics.ndcg import ndcg_metric
from metrics.novelty import novelty
from metrics.diversity import diversity
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from collections import defaultdict

# @TRAIN FUNCTION 
def train(**kwargs):

    # kwargs = {"argumento": "value"}
    # kwargs = [(arg, value), (arg2, value2)...]

    # DATASET LOADING
    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    opt.parse(kwargs) 

    # PARALLEL CONFIG
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # IMPORT MODEL
    if 'model' not in kwargs:
        raise Exception("Model not provided.")
    
    if opt.bert == "zeroshot":
        opt.model = f"{opt.model}_ZeroShot"
        model = Model(opt, getattr(models, opt.model))
    elif opt.bert == "finetunning":
        opt.model = f"{opt.model}_FineTunning"
        model = Model(opt, getattr(models, opt.model))
    else:
        model = Model(opt, getattr(models, opt.model))
    print(f"Model: {opt.model}")

    # SENDING MODEL TO GPU, IF EXISTS
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    # EXCEPTION FEATURES ERROR
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
    
    # LOAD INTERACTIONS
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    # OPTMIZERS
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    if opt.transnet:
        
        target_optim = optim.Adam(model.net.target_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        target_sr = optim.lr_scheduler.StepLR(target_optim, step_size=5, gamma=0.8)

        trans_optim = optim.Adam(model.net.source_net.trans_param(), lr=opt.lr, weight_decay=opt.weight_decay)
        trans_sr = optim.lr_scheduler.StepLR(trans_optim, step_size=5, gamma=0.8)

    # TRAINING STAGE
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):

        total_loss = 0.0
        total_maeloss = 0.0

        # MODEL IN TRAIN STAGE
        model.train()
        print(f"{now()}  Epoch {epoch}...")

        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
                
            # UNPACK NEDDED INPUTS
            train_datas = unpack_input(opt, train_datas)

            # RESET GRADIENT FOR BACKPROPAGATION
            optimizer.zero_grad()

            # PREDICT OUTPUT
            '''
                Transnet Operation
            '''
            if opt.transnet:

                l1, l2 = model(train_datas)
                s_lat, output, t_lat, t_pred = l1[0], l1[1], l2[0], l2[1]


                loss_t = F.l1_loss(t_pred, scores)
                target_optim.zero_grad()
                loss_t.backward()
                target_optim.step()

                loss_trans = F.mse_loss(s_lat, t_lat.detach())
                trans_optim.zero_grad()
                loss_trans.backward()
                trans_optim.step()

            else:

                output = model(train_datas)

            

            # CALCULATE LOSSES
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)

            # LOSS FUNCTION METHOD
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss

            # BACKPROPAGATION
            loss.backward()
            optimizer.step()

            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae, valoutput = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.emb_opt)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        # SMOOTHING BACKPROPAGATION
        scheduler.step()

        if opt.transnet:
            target_sr.step()
            trans_sr.step()
        
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")

        # VALIDATION TESTING FOR MODEL SAVE
        val_loss, val_mse, val_mae, val_prediction = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            model.save(name=opt.dataset, opt=opt.emb_opt)
            min_loss = val_loss
            print("model save")
        if val_mse < best_res:
            best_res = val_mse
        print("*"*30)

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.emb_opt} best_res:  {best_res}")
    print("----"*20)


# @TEST FUNCTION
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
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test dataset")

    predict_loss, test_mse, test_mae, test_prediction = predict(model, test_data_loader, opt)

    train_data = ReviewData(opt.data_root, mode="Train")
    val_data = ReviewData(opt.data_root, mode="Validation")

    user_rated_items = defaultdict(set)
    # Itens avaliados no treino
    for instance in train_data.x:
        user = int(instance[0].flatten()[0])
        item = instance[1]
        user_rated_items[user].add(item)
    # Itens avaliados na validação
    for instance in val_data.x:
        user = int(instance[0].flatten()[0])
        item = instance[1]
        user_rated_items[user].add(item)

     # --- Cálculo da Diversidade das Recomendações ---
    path = 'dataset/.data/' + f'{opt.dataset}_{opt.emb_opt}'
    with open(path + '/train/itemDoc2Index.npy', 'rb') as f:
        embeddings = np.load(f)

    rank_predictions = {}
    diversity_list = []

    test_users = set([int(x[0].flatten()[0]) for x in test_data.x])
    
    #user = list(test_users)[0]
    for user in tqdm(test_users, desc="Processando usuários"):
        rated_items = user_rated_items.get(user, set())
        candidate_items = list(set(range(opt.item_num - 2)) - rated_items)
        test_user_data = unpack_input(opt, zip([user] * len(candidate_items), candidate_items))
        all_predictions = []
        # Processa os itens em batches de tamanho opt.batch_size
        for i in range(0, len(candidate_items), opt.batch_size):
            batch_items = candidate_items[i : i + opt.batch_size]
            test_user_data = unpack_input(opt, zip([user] * len(batch_items), batch_items))
            
            if opt.transnet:
                output, _ = model(test_user_data)
                output = output[1]
            else:
                output = model(test_user_data)
            
            batch_predictions = output.cpu().tolist()
            all_predictions.extend(batch_predictions)

        item_predictions = list(zip(candidate_items, all_predictions))
        item_predictions = sorted(item_predictions, key=lambda x: x[1], reverse=True)

        top_k = 500  
        top_k_dict = {item: score for item, score in item_predictions[:top_k]}
        rank_predictions[user] = top_k_dict


    recommendations_file = f"results/{opt.dataset}_{opt.model}_top500_recommendations.json"
    with open(recommendations_file, "w") as f:
        json.dump(rank_predictions, f, indent=4)
    print(f"Top500 recomendações salvas em {recommendations_file}")
    
# @EVALUATE
def evaluate(**kwargs):
    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    
    # Se você precisar ajustar o nome do modelo, conforme já feito na função test:
    if opt.bert == "zeroshot":
        opt.model = f"{opt.model}_ZeroShot"
    elif opt.bert == "finetunning":
        opt.model = f"{opt.model}_FineTunning"
    
    # Caminho para o arquivo de recomendações já gerado
    recommendations_file = f"results/{opt.dataset}_{opt.model}_top10_recommendations.json"
    with open(recommendations_file, "r") as f:
         rank_predictions = json.load(f)
    print(f"Recomendações carregadas de {recommendations_file}")

    # Carrega os embeddings
    emb_path = f"dataset/.data/{opt.dataset}_{opt.emb_opt}/train/itemDoc2Index.npy"
    with open(emb_path, "rb") as f:
         embeddings = np.load(f)
    print(f"Embeddings carregados de {emb_path}")

    # Cálculo da diversidade
    avg_diversity = diversity(rank_predictions, embeddings)
    print(f"Diversidade global: {avg_diversity:.2f}")

    # Reconstrói o ground truth a partir dos dados de teste
    test_data = ReviewData(opt.data_root, mode="Test")
    test_users = set([int(x[0].flatten()[0]) for x in test_data.x])
    inner_ground_truth = {user: {} for user in test_users}
    for interaction in test_data.x:
        user = int(interaction[0].flatten()[0])
        item = int(interaction[0].flatten()[1])
        rating = float(interaction[1])
        inner_ground_truth[user][item] = rating

    test_ground_truth = {
        "feedback": inner_ground_truth,
        "users": list(inner_ground_truth.keys()),
        "items": list({item for user in inner_ground_truth for item in inner_ground_truth[user]})
    }
    rank_predictions = {
        int(user): {int(item): score for item, score in recs.items()}
        for user, recs in rank_predictions.items()
}
    
    rank_evaluator = RatingPredictionEvaluation(sep='\t', n_rank=[10], as_rank=True, metrics=['PREC'])
    rank_metrics = rank_evaluator.evaluate(rank_predictions, test_ground_truth)

    for metric_name, metric_value in rank_metrics.items():
        print(f"{metric_name}: {metric_value}")

    print(f"Precision: {rank_metrics.get('PREC@10'):.4f}")

    # Cria o dicionário de resultados e salva em um arquivo JSON
    result = {
       "dataset": opt.dataset,
       "model": opt.model,
       "diversity": avg_diversity,
       "precision": rank_metrics.get('PREC@10'),
       "recall": rank_metrics.get('RECALL@10')
    }
    output_file = f"results/{opt.dataset}_{opt.model}_results_ajustado.json"
    with open(output_file, "w") as f:
       json.dump(result, f, indent=4)
    print(f"Resultados salvos em {output_file}")

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

            mse_loss = torch.mean((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.mean(abs(output-scores))
            total_maeloss += mae_loss.item()

            rmse, precision, recall = calculate_metrics(scores, output)
            novel = novelty(scores.cpu().tolist(), output.cpu().tolist())
            #diver = diversity(scores.cpu().tolist(), output.cpu().tolist())

            mse_values.append(mse_loss.cpu().item())
            rmse_values.append(rmse.item())
            mae_values.append(mae_loss.cpu().item())
            #precision_values.append(precision.item())
            #recall_values.append(recall.item())
            novelty_values.append(novel)
            #diversity_values.append(diver)

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

    # if opt.diversity_metrics:
    #     ...
        # Preciso das categorias dos itens
        # Preciso de listas de recomendação e listas verdadeiras
        # Calculo métricas de diversidade

    
    if opt.statistical_test:

        df_error = {
            "mse":mse_values, 
            "mae":mae_values,
            "rmse": rmse_values,
            #"precision": precision_values,
            #"recall": recall_values,
            #"diversity": diversity_values,
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
                NOVELTY mean: {np.array(novelty_values).mean():.2f}'''
                #PRECISION mean: {np.array(precision_values).mean():.2f},
                #RECALL mean: {np.array(recall_values).mean():.2f},
                #DIVERSITY mean: {np.array(diversity_values).mean():.2f}'''
                
            )

            

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    
    # RETURN TO TRAIN STAGE
    model.train()

    return total_loss, mse, mae, output


# @UNPACK INTERACTION FEATURES
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

def extract_tensor_value(tensor_str):
    return tensor_str.split(",")[0].replace("tensor(", "").strip()

if __name__ == "__main__":
    fire.Fire()