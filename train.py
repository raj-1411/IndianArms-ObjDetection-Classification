import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import argparse


class Trainer:
    def __init__(self, data_dir, n_splits=5, project_name="KFold_Uniform_Detection"):
        """
        Initialize K-Fold validator for YOLO

        Args:
            data_dir: Path to directory containing train/images and train/labels
            n_splits: Number of folds for cross-validation
            project_name: Name for the project directory
        """
    
        self.n_splits = n_splits
        self.project_name = project_name
        self.fold_base_dir = Path(data_dir)


    def _print_class_distribution(self, label_dir, split_name):
        """Print class distribution for a dataset split"""
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        total_annotations = 0

        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_annotations += 1

        print(f"  {split_name} distribution:")
        class_names = ['BSF', 'CRPF', 'JK', 'BG']
        for cls_id in range(4):
            count = class_counts.get(cls_id, 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"    {class_names[cls_id]}: {count} ({percentage:.1f}%)")



    def train_fold(self, fold_num, yaml_path, config=None):
        """
        Train YOLO model on a specific fold
        """
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_num}/{self.n_splits}")
        print(f"{'='*50}")

        
        default_config = {
            'data': str(yaml_path),
            'epochs': 200,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.001,
            'lrf': 0.01,
            'warmup_epochs': 5.0,
            'box': 1.5,
            'cls': 2.0,
            'dfl': 1.0,
            'augment': True,
            'hsv_h': 0.01,
            'hsv_s': 0.4,
            'hsv_v': 0.3,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.3,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.15,
            'patience': 50,
            'close_mosaic': 30,
            'optimizer': 'AdamW',
            'freeze': 0,
            'pretrained': True,
            'val': True,
            'plots': True,
            'save': True,
            'project': self.project_name,
            'name': f'fold_{fold_num}',
            'exist_ok': True,
            'cache': 'ram'
        }

        
        if config:
            default_config.update(config)

        
        model = YOLO("yolo11m.pt")
        results = model.train(**default_config)

        
        val_results = model.val(conf=0.3, iou=0.6)

        return {
            'fold': fold_num,
            'model': model,
            'train_results': results,
            'val_results': val_results,
            'best_model_path': Path(self.project_name) / f'fold_{fold_num}' / 'weights' / 'best.pt'
        }

    def run_kfold_validation(self, config=None):
        """
        Run complete K-fold cross-validation
        """
        
        all_results = []


        for fold_num in range(1, self.n_splits + 1):

            fold_dir = self.fold_base_dir / f"fold_{fold_num}"
            yaml_path = fold_dir / 'data.yaml'

            train_img_dir = fold_dir / "train" / "images"
            train_lbl_dir = fold_dir / "train" / "labels"
            val_img_dir = fold_dir / "val" / "images"
            val_lbl_dir = fold_dir / "val" / "labels"

            print(f"\nFold {fold_num} created:")
            print(f"  Training samples: {len(os.listdir(train_img_dir))}")
            print(f"  Validation samples: {len(os.listdir(val_img_dir))}")

            self._print_class_distribution(train_lbl_dir, "Training")
            self._print_class_distribution(val_lbl_dir, "Validation")

            
            fold_results = self.train_fold(fold_num, yaml_path, config)
            all_results.append(fold_results)

        
            import torch
            torch.cuda.empty_cache()

        
        self.aggregate_results(all_results)

        return all_results

    def aggregate_results(self, all_results):
        """
        Aggregate and display results from all folds
        """
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION RESULTS ({self.n_splits} folds)")
        print(f"{'='*60}")

        
        metrics_data = []

        for result in all_results:
            fold_num = result['fold']
            val_results = result['val_results']

            
            metrics = {
                'Fold': fold_num,
                'mAP50': val_results.box.map50 if hasattr(val_results.box, 'map50') else 0,
                'mAP50-95': val_results.box.map if hasattr(val_results.box, 'map') else 0,
                'Precision': val_results.box.mp if hasattr(val_results.box, 'mp') else 0,
                'Recall': val_results.box.mr if hasattr(val_results.box, 'mr') else 0,
            }

            
            if hasattr(val_results.box, 'ap50'):
                class_names = ['BSF', 'CRPF', 'JK', 'BG']
                for i, name in enumerate(class_names):
                    if i < len(val_results.box.ap50):
                        metrics[f'AP50_{name}'] = val_results.box.ap50[i]

            metrics_data.append(metrics)

        
        df = pd.DataFrame(metrics_data)

        
        print("\nPer-Fold Results:")
        print(df.to_string(index=False))

        
        print("\nOverall Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Fold']

        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col:20s}: {mean_val:.4f} ± {std_val:.4f}")

        
        results_path = Path(self.project_name) / 'kfold_results.csv'
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        
        best_fold = df.loc[df['mAP50-95'].idxmax()]
        print(f"\nBest Fold: {int(best_fold['Fold'])} with mAP50-95: {best_fold['mAP50-95']:.4f}")
        print(f"Best model saved at: {self.project_name}/fold_{int(best_fold['Fold'])}/weights/best.pt")

        return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run K-Fold Cross-Validation for YOLO")
    parser.add_argument('--fold_base_dir', type=str, default="dataset_kfold_5", help="Base directory for K-Fold dataset")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--project_name', type=str, default="UniformDetection_KFold", help="Name for the project directory")
    parser.add_argument('--fold_exist', type=bool, default=False, help="Whether folds exist, otherwise the user has to generate folds using KFold_gen.py")
    
    args = parser.parse_args()
    if args.fold_exist:
        obj = Trainer(
            data_dir=args.fold_base_dir,
            n_splits=args.n_splits,
            project_name=args.project_name
        )


        custom_config = {
            'epochs': 250,       
            'batch': 16,         
            'imgsz': 640,
            'patience': 100,

        }

        print("Starting K-Fold Training...")
        results = obj.run_kfold_training(config=custom_config)

        print("\nK-Fold Training Complete!")

    else:
        print("Please generate folds using KFold_gen.py before running this script.")
        print("Exiting...")
