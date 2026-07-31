#%%
# Proposed framework: jointly trains the Hawkes popularity model and the density-ratio
# recommender (MF/GRU-style backbones), following Algorithm 1 in the paper. Negatives for the
# user loss are drawn from the current Hawkes snapshot instead of uniformly, and the popularity
# model is fit on uniformly-sampled negatives via the sampled-softmax objective ell_pop.
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from module.dataset import UserItemTime
from module.procedure import computeTopNAccuracy
from module.model import score_pair, score_all, MODEL_REGISTRY
from module.utils import parse_args, set_seed, set_device, get_epoch
from module.sampler import make_prior_snapshot, sample_epoch_negatives
from module.debias import build_debias_model


#%%
args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)


#%%
dataset = UserItemTime(args.data_path, args.dataset, args.time_unit, 50, args.max_seq_len)

# item_time_array is stored in raw timestamp units; rescale it to match args.time_unit
if args.time_unit == "s":
    pass
elif args.time_unit == "m":
    dataset.item_time_array = dataset.item_time_array / 60
elif args.time_unit == "h":
    dataset.item_time_array = dataset.item_time_array / 60 / 60
elif args.time_unit == "d":
    dataset.item_time_array = dataset.item_time_array / 60 / 60 / 24

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1
hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)
all_item_idxs = np.arange(dataset.m_item)
epoch = 0


#%%
model_name = getattr(args, "model_name", "mf").lower()
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
model_class = MODEL_REGISTRY[model_name]
# wrap the chosen backbone with the Hawkes popularity head (mu/alpha/beta networks)
debiased_class = build_debias_model(model_class)
model = debiased_class(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    tau=args.tau,
    depth=args.depth,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    dropout=args.dropout,
    ).to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.decay,
)

save_dir = Path(args.save_path)
pattern = f"proposed_{args.model_name}_gamma{args.gamma}_e???_seed{args.seed}.pt"
matched_files = sorted(save_dir.glob(pattern))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("MODEL LOADED!")


#%%
# initial Hawkes snapshot and popularity-aware negatives for the density-ratio loss
dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)
snapshot = make_prior_snapshot(model)
hot_negs = sample_epoch_negatives(
    snapshot=snapshot,
    train_events=dataset.train_hot_events,
    num_items=dataset.m_item,
    num_negatives=args.contrast_size-1,
)
cold_negs = sample_epoch_negatives(
    snapshot=snapshot,
    train_events=dataset.train_cold_events,
    num_items=dataset.m_item,
    num_negatives=args.contrast_size-1,
)


while epoch < args.epochs:
    epoch += 1
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_user_loss = 0.0
    epoch_item_loss = 0.0

    for idx in range(batch_num):
        user_loss = torch.zeros(1).to(args.device)
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch*idx : (idx + 1)*cold_mini_batch]

        """USER"""
        # density-ratio loss with popularity-aware negatives (Eq. 5), weighted by gamma
        hot_anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        cold_anchor_user = torch.tensor(dataset.cold_user_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_pos_item = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)

        anchor_user = torch.cat([cold_anchor_user, hot_anchor_user], dim=0)
        pos_item = torch.cat([cold_pos_item, hot_pos_item], dim=0)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        hot_neg_item = torch.tensor(hot_negs[hot_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item = torch.tensor(cold_negs[cold_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.cat([cold_neg_item, hot_neg_item], dim=0)

        pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
        neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)
        user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean() * args.gamma
        epoch_user_loss += user_loss.item()


        """ITEM"""
        # sampled-softmax popularity loss ell_pop (Eq. under Estimating Time-Varying Item
        # Popularity): uses fresh uniform negatives, independent of the density-ratio negatives
        hot_neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.cat([cold_neg_item, hot_neg_item], dim=0)

        hot_pos_time = torch.Tensor(dataset.hot_event_time_list[hot_sample_idx]).reshape(-1, 1, 1).to(args.device)
        cold_pos_time = torch.Tensor(dataset.cold_event_time_list[cold_sample_idx]).reshape(-1, 1, 1).to(args.device)
        pos_time = torch.cat([cold_pos_time, hot_pos_time], dim=0)

        hot_pos_time_all = torch.Tensor(dataset.hot_pos_time_all[hot_sample_idx]).to(args.device)
        cold_pos_time_all = torch.Tensor(dataset.cold_pos_time_all[cold_sample_idx]).to(args.device)
        pos_time_all = torch.cat([cold_pos_time_all, hot_pos_time_all], dim=0)

        hot_neg_time_all = torch.Tensor(dataset.hot_neg_time_all[hot_sample_idx]).to(args.device)
        cold_neg_time_all = torch.Tensor(dataset.cold_neg_time_all[cold_sample_idx]).to(args.device)
        neg_time_all = torch.cat([cold_neg_time_all, hot_neg_time_all], dim=0)

        batch_items = torch.concat([pos_item.unsqueeze(-1), neg_item], -1).reshape(pos_item.shape[0], -1)
        batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

        # lambda_v(t) for the positive item followed by its negatives; index 0 is always the positive
        logits = model.prior(batch_items, pos_time, batch_time_all)
        log_logits = torch.log(logits + 1e-9)
        item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean() * (1-args.gamma)
        epoch_item_loss += item_loss.item()

        dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)


        # combined objective L = gamma * ell_DR + (1 - gamma) * ell_pop
        total_loss = item_loss + user_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss / batch_num:.4f} / item: {epoch_item_loss / batch_num:.4f}")


    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_user_loss,
        }, f"{args.save_path}/proposed_{args.model_name}_gamma{args.gamma}_e{epoch}_seed{args.seed}.pt")


    if epoch % args.pair_reset_interval == 0:
        # refresh the Hawkes snapshot and resample popularity-aware negatives (snapshot reset in Algorithm 1)
        print("Reset Negs")
        snapshot = make_prior_snapshot(model)
        hot_negs = sample_epoch_negatives(
            snapshot=snapshot,
            train_events=dataset.train_hot_events,
            num_items=dataset.m_item,
            num_negatives=args.contrast_size-1,
        )
        cold_negs = sample_epoch_negatives(
            snapshot=snapshot,
            train_events=dataset.train_cold_events,
            num_items=dataset.m_item,
            num_negatives=args.contrast_size-1,
        )


    if epoch % args.evaluate_interval == 0:
        pred_list, gt_list = [], []
        model.eval()
        with torch.no_grad():
            mu, alpha, beta = model.prior_parameters_from_embeddings()

        for (user, item), pos_time_val in dataset.valid_user_item_time.items():
            hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                resid = score_all(model, hist_item_t, user_t).squeeze(0)

            # per-item Hawkes log-intensity at the query time, normalized into a log-probability
            pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)
            item_logits_list = []
            for idx2 in range(dataset.m_item // args.batch_size + 1):
                item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
                if len(item_idx) == 0:
                    continue

                batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
                batch_time_mask = batch_time_all < pos_time_t
                batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)
                time_intensity = (torch.exp(-beta * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (mu[item_idx] + alpha[item_idx] * time_intensity.squeeze(-1)).flatten()
                item_logits_list.append(logits)

            item_logits = torch.concat(item_logits_list)
            item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)
            # ranking score r_eta = eta * log popularity + (1 - eta) * debiased score (inference calibration)
            pred = (item_log_prob * args.eta + resid * (1-args.eta)).cpu()

            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        print(dict(zip([f"valid_precision_{k}_{epoch}" for k in args.topks], valid_results[0])))
        print(dict(zip([f"valid_recall_{k}_{epoch}" for k in args.topks], valid_results[1])))
        print(dict(zip([f"valid_ndcg_{k}_{epoch}" for k in args.topks], valid_results[2])))
        print(dict(zip([f"valid_mrr_{k}_{epoch}" for k in args.topks], valid_results[3])))


pred_list, gt_list = [], []
model.eval()
with torch.no_grad():
    mu, alpha, beta = model.prior_parameters_from_embeddings()

for (user, item), pos_time_val in dataset.test_user_item_time.items():
    hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    user_t = torch.tensor([user], dtype=torch.long, device=args.device)

    with torch.no_grad():
        resid = score_all(model, hist_item_t, user_t).squeeze(0)

    pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)
    item_logits_list = []
    for idx2 in range(dataset.m_item // args.batch_size + 1):
        item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
        if len(item_idx) == 0:
            continue

        batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
        batch_time_mask = batch_time_all < pos_time_t
        batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)
        time_intensity = (torch.exp(-beta * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
        logits = (mu[item_idx] + alpha[item_idx] * time_intensity.squeeze(-1)).flatten()
        item_logits_list.append(logits)

    item_logits = torch.concat(item_logits_list)
    item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)
    pred = (item_log_prob * args.eta + resid * (1-args.eta)).cpu()

    exclude_items = list(dataset._allPos[user])
    pred[exclude_items] = -9999
    _, pred_k = torch.topk(pred, k=max(args.topks))
    pred_list.append(pred_k.cpu())
    gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

print(dict(zip([f"test_precision_{k}_{epoch}" for k in args.topks], test_results[0])))
print(dict(zip([f"test_recall_{k}_{epoch}" for k in args.topks], test_results[1])))
print(dict(zip([f"test_ndcg_{k}_{epoch}" for k in args.topks], test_results[2])))
print(dict(zip([f"test_mrr_{k}_{epoch}" for k in args.topks], test_results[3])))


if args.debiased_eval == "true":
    # additional breakdown by head/tail items under different popularity-window definitions (Fig. 3)
    eval_datasets = [
        ("head_overall", dataset.test_head_overall_dict),
        ("head_recent_3d", dataset.test_head_recent_3d_dict),
        ("head_recent_7d", dataset.test_head_recent_7d_dict),
        ("tail_overall", dataset.test_tail_overall_dict),
        ("tail_recent_3d", dataset.test_tail_recent_3d_dict),
        ("tail_recent_7d", dataset.test_tail_recent_7d_dict),
    ]


    for (split_name, data_split) in eval_datasets:
        pred_list, gt_list = [], []
        model.eval()

        for (user, item), pos_time_val in dataset.set_to_pair(data_split, dataset.time_dict, dataset.time_unit).items():
            hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                resid = score_all(model, hist_item_t, user_t).squeeze(0)

            pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)
            item_logits_list = []

            for idx2 in range(dataset.m_item // args.batch_size + 1):
                item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
                if len(item_idx) == 0:
                    continue

                batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
                batch_time_mask = batch_time_all < pos_time_t
                batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)
                time_intensity = (torch.exp(-beta * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (mu[item_idx] + alpha[item_idx] * time_intensity.squeeze(-1)).flatten()
                item_logits_list.append(logits)

            item_logits = torch.concat(item_logits_list)
            item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)
            pred = (item_log_prob * args.eta + resid * (1-args.eta)).cpu()

            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        print(dict(zip([f"test_{split_name}_precision_{k}_{epoch}" for k in args.topks], test_results[0])))
        print(dict(zip([f"test_{split_name}_recall_{k}_{epoch}" for k in args.topks], test_results[1])))
        print(dict(zip([f"test_{split_name}_ndcg_{k}_{epoch}" for k in args.topks], test_results[2])))
        print(dict(zip([f"test_{split_name}_mrr_{k}_{epoch}" for k in args.topks], test_results[3])))
