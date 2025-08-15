import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import KFold
import argparse


class YOLOKFoldValidator:
    def __init__(self, data_dir, n_splits=5):
        """
        Initialize K-Fold validator for YOLO

        Args:
            data_dir: Path to directory containing train/images and train/labels
            n_splits: Number of folds for cross-validation
            project_name: Name for the project directory
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "train" / "images"
        self.labels_dir = self.data_dir / "train" / "labels"
        self.n_splits = n_splits
    
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = [self.labels_dir / f"{img.stem}.txt" for img in self.image_files]

        print(f"Found {len(self.image_files)} images for {n_splits}-fold cross-validation")

        
        self.fold_base_dir = Path(f"dataset_kfold_{n_splits}")
        self.fold_base_dir.mkdir(exist_ok=True)

    def create_fold_split(self, train_indices, val_indices, fold_num):
        """
        Create train/val split for a specific fold
        """
        fold_dir = self.fold_base_dir / f"fold_{fold_num}"

        
        train_img_dir = fold_dir / "train" / "images"
        train_lbl_dir = fold_dir / "train" / "labels"
        val_img_dir = fold_dir / "val" / "images"
        val_lbl_dir = fold_dir / "val" / "labels"

        for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
            for file in dir_path.glob("*"):
                file.unlink()

        
        for idx in train_indices:
            img_src = self.image_files[idx]
            lbl_src = self.label_files[idx]

            shutil.copy2(img_src, train_img_dir / img_src.name)
            if lbl_src.exists():
                shutil.copy2(lbl_src, train_lbl_dir / lbl_src.name)

        
        for idx in val_indices:
            img_src = self.image_files[idx]
            lbl_src = self.label_files[idx]

            shutil.copy2(img_src, val_img_dir / img_src.name)
            if lbl_src.exists():
                shutil.copy2(lbl_src, val_lbl_dir / lbl_src.name)

        
        data_yaml = {
            'path': str(fold_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 4,  # 4 classes including background
            'names': ['BSF', 'CRPF', 'JK', 'BG']
        }

        yaml_path = fold_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        
        print(f"\nFold {fold_num} created:")
        print(f"  Training samples: {len(train_indices)}")
        print(f"  Validation samples: {len(val_indices)}")

        
        self._print_class_distribution(train_lbl_dir, "Training")
        self._print_class_distribution(val_lbl_dir, "Validation")


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


    def create_kfold(self):
        """
        Run complete K-fold cross-validation
        """
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        
        for fold_num, (train_indices, val_indices) in enumerate(kfold.split(self.image_files), 1):
            self.create_fold_split(train_indices, val_indices, fold_num)

        

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="Run K-Fold Cross-Validation for YOLO")
    argument_parser.add_argument('--data_dir', type=str, default="dataset", help="Path to dataset directory")
    argument_parser.add_argument('--n_splits', type=int, default=5, help="Number of folds for cross-validation")
    
    args = argument_parser.parse_args()
    
    validator = YOLOKFoldValidator(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
    )

    create_kfold = validator.create_kfold()
    print(f"K-Fold cross-validation created with {validator.n_splits} folds.")
