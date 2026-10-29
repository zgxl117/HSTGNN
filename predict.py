import os
import torch
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from torch_geometric.loader import DataLoader
from mydatasets.customized_dataset import CustomGraphDataset, DisplayDatasetInfo
from train import GNNLightning
from models.models import get_model

from utils.utils import get_logger
from utils.config import load_config

import pandas as pd


logger = get_logger('FsimNNPrediction')


def evaluate_model(model, test_loader, output_path, num_classes=5):
    
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_path, exist_ok=True)
    
    results = []
    all_preds =[]
    all_labels = []

    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
    recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    per_class_precision = Precision(task="multiclass", num_classes=num_classes, average=None)
    per_class_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
    per_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)

    with torch.no_grad():  
        for data in test_loader:
            data = data.to(model.device)
            outputs = model(data)
            # outputs: [N, 2]
            p = torch.softmax(outputs, dim=1)   # 如果你已经sigmoid过，这里也能看相对大小
            print("mean prob class1:", p[:,1].mean().item())
            print("max  prob class1:", p[:,1].max().item())
            print("fraction class1>class0:", (p[:,1] > p[:,0]).float().mean().item())
            preds = torch.argmax(outputs, dim=1)

            test_mask = data["ff"].test_mask
            if test_mask.sum() > 0:
                preds = preds[test_mask].cpu().numpy()
                y = data["ff"].y[test_mask].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y)
               
                node_ids = data["ff"].node_ids[test_mask].cpu().numpy()
                graph_id = int(data.graph_id.reshape(-1)[0].item())
                source_path = data.source_path[0] if isinstance(data.source_path, (list, tuple)) else data.source_path
                for i, node_id in enumerate(node_ids):
                    results.append({
                        "source_path": source_path,
                        "graph_id": graph_id,
                        "node_id": node_id,
                        "predicted_label": preds[i],
                        "true_label": y[i]
                    })



    # Save results to a CSV file
    output_file = os.path.join(output_path,'results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    # Convert lists to tensors before computing metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    print("label bincount:", torch.bincount(all_labels, minlength=num_classes))
    print("pred  bincount:", torch.bincount(all_preds,  minlength=num_classes))

    # Compute metrics
    acc = accuracy(all_preds, all_labels)
    prec = precision(all_preds, all_labels)
    rec = recall(all_preds, all_labels)
    f1 = f1_score(all_preds, all_labels)
    class_prec = per_class_precision(all_preds, all_labels)
    class_rec = per_class_recall(all_preds, all_labels)
    class_f1 = per_class_f1(all_preds, all_labels)
    label_counts = torch.bincount(all_labels, minlength=num_classes)
    pred_counts = torch.bincount(all_preds, minlength=num_classes)
    
    metrics_df = pd.DataFrame({
        "Accuracy": [acc.item()],
        "Macro_Precision": [prec.item()],
        "Macro_Recall": [rec.item()],
        "Macro_F1": [f1.item()]
    })
        
    # Save metrics to a CSV file
    scores_file = os.path.join(output_path,'scores.csv')
    metrics_df.to_csv(scores_file, index=False)
    logger.info(f"Metrics saved to {scores_file}")    

    per_class_rows = []
    for class_id in range(num_classes):
        per_class_rows.append({
            "class_id": class_id,
            "support": label_counts[class_id].item(),
            "pred_count": pred_counts[class_id].item(),
            "precision": class_prec[class_id].item(),
            "recall": class_rec[class_id].item(),
            "f1": class_f1[class_id].item(),
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_file = os.path.join(output_path, "per_class_scores.csv")
    per_class_df.to_csv(per_class_file, index=False)
    logger.info(f"Per-class metrics saved to {per_class_file}")

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Macro Precision: {prec:.4f}")
    logger.info(f"Macro Recall: {rec:.4f}")
    logger.info(f"Macro F1: {f1:.4f}")
    for row in per_class_rows:
        logger.info(
            "Class %s | support=%s pred_count=%s precision=%.4f recall=%.4f f1=%.4f",
            row["class_id"],
            row["support"],
            row["pred_count"],
            row["precision"],
            row["recall"],
            row["f1"],
        )

    return acc, prec, rec, f1




def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)   

    dataset_config = config["dataset"]
    root_dir = dataset_config["root_dir"]
    train_val_dirs = dataset_config["train_val_dirs"]
    test_dirs = dataset_config["test_dirs"]
    split_mode = dataset_config.get("split_mode", "spatio")
    if split_mode in {"feature_coverage", "grouped_ff_feature_coverage"}:
        split_config = dataset_config.get(split_mode, {})
    else:
        split_config = dataset_config.get("split_config", {})

    model_config = config["model"]
    edge_vector_size = model_config.get("edge_vector_size", 10)
    time_window_size = model_config.get("time_window_size", 60)    
    num_classes = model_config.get("num_classes", 5)
    
    test_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, split_config=split_config, phase="test", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    DisplayDatasetInfo(test_dataset, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
    
    
    best_model_path = os.path.join(config["model_path"], "best_model.ckpt") 
    logger.info(f"Best model is read from: {best_model_path}")
    
    model = get_model(config)
    model = GNNLightning.load_from_checkpoint(best_model_path, model=model)        
                      
    output_path = config["output_path"]
    evaluate_model(model, test_loader, output_path, num_classes=num_classes)




if __name__ == '__main__':
    main()
