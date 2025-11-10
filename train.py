"""
Training script for PointNet++ Binary Classification of Rock Joints
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json

from pointnet2_model import PointNet2Classification, PointNet2Segmentation
from dataset import RockJointDataset


class Trainer:
    """
    Trainer class for PointNet++ rock joint classification
    """
    def __init__(self, model, train_loader, test_loader, config):
        """
        Args:
            model: PointNet2Classification model
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            config: Dict with training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

        # Loss function
        if config.get('class_weights') is not None:
            weights = torch.FloatTensor(config['class_weights']).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_rate']
        )

        # Tracking - expanded to include precision/recall
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.train_precisions = []
        self.train_recalls = []
        self.test_precisions = []
        self.test_recalls = []
        self.best_test_acc = 0.0
        self.best_model_path = None

        # Use run_dir instead of save_dir
        self.save_dir = config['run_dir']

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch_data in enumerate(pbar):
          if len(batch_data) == 3:
              points, labels, indices = batch_data
          else:
              points, labels = batch_data
              indices = None

          points = points.to(self.device)  # [B, N, 3]
          labels = labels.to(self.device)  # [B, N]  ← per-point labels

          self.optimizer.zero_grad()

          # Forward pass
          outputs = self.model(points)  # [B, num_classes, N]

          # Reshape for loss computation
          outputs = outputs.permute(0, 2, 1).contiguous()  # [B, N, num_classes] # do we need this line? wouldnt view(-1,C) result in (BN,C) ?

          outputs = outputs.view(-1, self.config['num_classes'])  # [B*N, num_classes]
          labels = labels.view(-1)  # [B*N]

          # Compute loss on all points (3-class: all points have valid labels 0, 1, 2)
          loss = self.criterion(outputs, labels)

          # Backward pass
          loss.backward()
          self.optimizer.step()

          # Track metrics
          total_loss += loss.item()
          preds = torch.argmax(outputs, dim=1)  # [B*N]
          all_preds.extend(preds.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

          # Update progress bar
          pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 3-CLASS: All points have valid labels (0, 1, 2), no filtering needed
        accuracy = accuracy_score(all_labels, all_preds)

        # Use macro averaging for multi-class (equal weight to each class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Compute per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        return avg_loss, accuracy, precision, recall, all_labels, all_preds, per_class_precision, per_class_recall, per_class_f1

    def test_epoch(self, epoch):
        """Test for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {epoch} [Test]')
            for batch_idx, batch_data in enumerate(pbar):
              if len(batch_data) == 3:
                points, labels, indices = batch_data
              else:
                points, labels = batch_data
                indices = None

              points = points.to(self.device)  # [B, N, 3]
              labels = labels.to(self.device)  # [B, N]

              outputs = self.model(points)  # [B, num_classes, N]

              # Reshape for loss
              outputs = outputs.permute(0, 2, 1).contiguous()  # [B, N, num_classes]
              outputs = outputs.view(-1, self.config['num_classes'])  # [B*N, num_classes]
              labels_flat = labels.view(-1)  # [B*N]

              loss = self.criterion(outputs, labels_flat)
              total_loss += loss.item()

              # Get predictions and probabilities
              probs = torch.softmax(outputs, dim=1)  # [B*N, num_classes]
              preds = torch.argmax(outputs, dim=1)  # [B*N]

              all_preds.extend(preds.cpu().numpy())
              all_labels.extend(labels_flat.cpu().numpy())
              all_probs.extend(probs.cpu().numpy())

              pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.test_loader)

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 3-CLASS: All points have valid labels (0, 1, 2), no filtering needed
        accuracy = accuracy_score(all_labels, all_preds)

        # Use macro averaging for multi-class (equal weight to each class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Compute per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds, per_class_precision, per_class_recall, per_class_f1

    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("="*50 + "\n")

        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print("-" * 50)

            # Resample training data each epoch (if label sampling is enabled)
            # This uses cached KNN indices - no recomputation needed
            if hasattr(self.train_loader.dataset, 'resample_active_indices'):
                self.train_loader.dataset.resample_active_indices(epoch)

            # Train
            train_results = self.train_epoch(epoch)
            train_loss, train_acc, train_precision, train_recall, train_labels, train_preds, \
            train_per_class_precision, train_per_class_recall, train_per_class_f1 = train_results
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)

            # Test
            test_results = self.test_epoch(epoch)
            test_loss, test_acc, test_precision, test_recall, f1, test_labels, test_preds, \
            test_per_class_precision, test_per_class_recall, test_per_class_f1 = test_results
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            self.test_precisions.append(test_precision)
            self.test_recalls.append(test_recall)

            # Update learning rate
            self.scheduler.step()

            # Print macro-averaged metrics
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch} RESULTS (Macro-Averaged)")
            print(f"{'='*70}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f} | "
                  f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {f1:.4f}")

            # Print per-class metrics for both train and test
            # Get class names based on num_classes
            num_classes = self.config['num_classes']
            if num_classes == 2:
                class_names = ['No-Joint', 'Joint']
            elif num_classes == 3:
                class_names = ['Background', 'No-Joint', 'Joint']
            else:
                class_names = [f'Class {i}' for i in range(num_classes)]

            # Training set per-class metrics
            print(f"\n{'='*70}")
            print(f"PER-CLASS METRICS (Training Set)")
            print(f"{'='*70}")
            print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print(f"{'-'*70}")
            for i, class_name in enumerate(class_names):
                if i < len(train_per_class_precision):
                    print(f"{class_name:<15} {train_per_class_precision[i]:<12.4f} "
                          f"{train_per_class_recall[i]:<12.4f} {train_per_class_f1[i]:<12.4f}")
            print(f"{'='*70}")

            # Test set per-class metrics
            print(f"\n{'='*70}")
            print(f"PER-CLASS METRICS (Test Set)")
            print(f"{'='*70}")
            print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print(f"{'-'*70}")
            for i, class_name in enumerate(class_names):
                if i < len(test_per_class_precision):
                    print(f"{class_name:<15} {test_per_class_precision[i]:<12.4f} "
                          f"{test_per_class_recall[i]:<12.4f} {test_per_class_f1[i]:<12.4f}")
            print(f"{'='*70}")

            # Save best model
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_model_path = os.path.join(
                    self.save_dir,
                    f"best_model_acc{test_acc:.4f}.pth"
                )
                # Get voxel_size if available (for voxel dataset)
                voxel_size = None
                if hasattr(self.train_loader.dataset, 'get_voxel_size'):
                    voxel_size = self.train_loader.dataset.get_voxel_size()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'voxel_size': voxel_size,
                    'config': self.config
                }, self.best_model_path)
                print(f"  Saved best model: {self.best_model_path}")

            # Save checkpoint every N epochs
            if epoch % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.save_dir,
                    f"checkpoint_epoch{epoch}.pth"
                )

                # Get voxel_size if available (for voxel dataset)
                voxel_size = None
                if hasattr(self.train_loader.dataset, 'get_voxel_size'):
                    voxel_size = self.train_loader.dataset.get_voxel_size()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'voxel_size': voxel_size,
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses,
                    'train_accs': self.train_accs,
                    'test_accs': self.test_accs,
                    'train_precisions': self.train_precisions,
                    'train_recalls': self.train_recalls,
                    'test_precisions': self.test_precisions,
                    'test_recalls': self.test_recalls,
                    'config': self.config
                }, checkpoint_path)

                # Save confusion matrices at checkpoint
                print(f"Saving confusion matrices for epoch {epoch}...")
                self._plot_confusion_matrix(
                    train_labels, train_preds,
                    f'Train Confusion Matrix - Epoch {epoch}',
                    f'confusion_matrix_train_epoch{epoch}.png'
                )
                self._plot_confusion_matrix(
                    test_labels, test_preds,
                    f'Test Confusion Matrix - Epoch {epoch}',
                    f'confusion_matrix_test_epoch{epoch}.png'
                )

        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Test Accuracy: {self.best_test_acc:.4f}")
        print(f"Best Model: {self.best_model_path}")
        print("="*50 + "\n")

        # Final evaluation and visualizations
        self._final_evaluation(test_labels, test_preds)
        self._plot_training_curves()

    def _plot_confusion_matrix(self, labels, preds, title, filename):
        """Plot and save a confusion matrix"""
        cm = confusion_matrix(labels, preds)

        # Determine class labels based on number of classes
        num_classes = self.config['num_classes']
        if num_classes == 2:
            class_labels = ['No Joint', 'Joint']
        elif num_classes == 3:
            class_labels = ['Background', 'No Joint', 'Joint']
        else:
            class_labels = [f'Class {i}' for i in range(num_classes)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        cm_path = os.path.join(self.save_dir, filename)
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix: {cm_path}")

    def _final_evaluation(self, test_labels, test_preds):
        """Generate final evaluation metrics and confusion matrix"""
        # Confusion matrix
        self._plot_confusion_matrix(test_labels, test_preds,
                                    'Final Test Confusion Matrix',
                                    'confusion_matrix_final.png')

        # Classification report
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average=None, zero_division=0
        )

        # Build report dynamically based on number of classes
        num_classes = self.config['num_classes']
        class_names = ['background', 'no_joint', 'joint'] if num_classes == 3 else [f'class_{i}' for i in range(num_classes)]

        report = {}
        for i in range(num_classes):
            if i < len(precision):  # Check if class exists in predictions
                report[class_names[i]] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i])
                }
        report['overall_accuracy'] = float(self.best_test_acc)

        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Saved classification report: {report_path}")

    def _plot_training_curves(self):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
        axes[0, 0].plot(epochs, self.test_losses, label='Test Loss', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accs, label='Train Accuracy', linewidth=2, marker='o', markersize=3)
        axes[0, 1].plot(epochs, self.test_accs, label='Test Accuracy', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # Precision curves
        axes[1, 0].plot(epochs, self.train_precisions, label='Train Precision', linewidth=2, marker='o', markersize=3)
        axes[1, 0].plot(epochs, self.test_precisions, label='Test Precision', linewidth=2, marker='s', markersize=3)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Training and Test Precision', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # Recall curves
        axes[1, 1].plot(epochs, self.train_recalls, label='Train Recall', linewidth=2, marker='o', markersize=3)
        axes[1, 1].plot(epochs, self.test_recalls, label='Test Recall', linewidth=2, marker='s', markersize=3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('Training and Test Recall', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Saved training curves: {curves_path}")

        # Save metrics to CSV for further analysis
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': self.train_losses,
            'test_loss': self.test_losses,
            'train_acc': self.train_accs,
            'test_acc': self.test_accs,
            'train_precision': self.train_precisions,
            'test_precision': self.test_precisions,
            'train_recall': self.train_recalls,
            'test_recall': self.test_recalls
        })
        metrics_path = os.path.join(self.save_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved training metrics: {metrics_path}")

def main():
    """Main training function"""

    # =========================
    # Configuration
    # =========================
    config = {
        # Data paths
        'las_file': '/home/yush/local_backup_geotech/geotech_pointnet/data/w_E_p1_6cm_all.las',
        'no_joints_dxf': '/home/yush/local_backup_geotech/geotech_pointnet/data/no_joints.dxf',
        'joints_dxf': '/home/yush/local_backup_geotech/geotech_pointnet/data/polygon_joints.dxf',

        # Model parameters
        'input_channels': 3,  # XYZ only (set to 6 if using XYZ + RGB)
        'num_classes': 2,

        # Data parameters
        'patch_size': 2048,
        'normalize_mode': 'center',  # Options: 'none', 'center', 'center_scale'
        'augment_train': True,

        # Training parameters
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lr_decay_step': 20,
        'lr_decay_rate': 0.7,

        # Class balancing (optional)
        'class_weights': None,  # Set to [w0, w1] if needed, e.g., [1.0, 2.0]

        # Saving
        'save_dir': './checkpoints',
        'save_interval': 10,

        # Reproducibility
        'seed': 42
    }

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # =========================
    # Load Data
    # =========================
    print("Loading point cloud data...")
    import laspy
    import ezdxf
    import sys
    # sys.path.append('/home/claude')

    # Load LAS file
    las = laspy.read(config['las_file'])
    xyz_array = np.vstack([las.x, las.y, las.z]).transpose()

    # Initialize labels
    label_array = np.full(len(xyz_array), -1)

    # Load DXF files and extract polygons
    print("Loading polygon annotations...")
    from dataset import RockJointDataset  # Import function from your code

    # You need to add the lines_to_polygons function here or import it
    # For now, I'll assume you have the polygons already
    # Load polygons (you'll need to run your existing code)
    no_joints_dxf = ezdxf.readfile(config['no_joints_dxf'])
    joints_dxf = ezdxf.readfile(config['joints_dxf'])

    # Import your polygon extraction function
    from polygon_preprocessing import lines_to_polygons, label_points_in_polygon_2d_projection

    polygons_nojoint = lines_to_polygons(no_joints_dxf)
    polygons_joints = lines_to_polygons(joints_dxf)

    print(f"Found {len(polygons_nojoint)} no-joint polygons")
    print(f"Found {len(polygons_joints)} joint polygons")

    # Label points
    print("Labeling points within polygons...")
    label_array = label_points_in_polygon_2d_projection(
        xyz_array, label_array, polygons_nojoint, label_id=0
    )
    label_array = label_points_in_polygon_2d_projection(
        xyz_array, label_array, polygons_joints, label_id=1
    )

    # Create train/test split
    # NOTE: PLEASE GET RID OF THIS ASAP
    print("Creating train/test split...")
    point_a = np.array([464275, 9175697, 2546])
    point_b = np.array([464294, 9175699, 2552])

    full_ab = point_b[:2] - point_a[:2]
    full_ap = xyz_array[:, :2] - point_a[:2]
    full_cross_product = full_ap[:, 0] * full_ab[1] - full_ap[:, 1] * full_ab[0]

    full_train_mask = full_cross_product <= 0
    full_test_mask = full_cross_product > 0

    mask_where_labels_exist = label_array != -1
    train_mask = full_train_mask & mask_where_labels_exist
    test_mask = full_test_mask & mask_where_labels_exist

    print(f"Training points: {np.sum(train_mask)}")
    print(f"Test points: {np.sum(test_mask)}")

    # Prepare polygon dictionaries for dataset
    polygons_dict_train = {
        'label_0': [poly for poly in polygons_nojoint],  # Filter based on train_mask if needed
        'label_1': [poly for poly in polygons_joints]
    }

    polygons_dict_test = {
        'label_0': [poly for poly in polygons_nojoint],  # Filter based on test_mask if needed
        'label_1': [poly for poly in polygons_joints]
    }

    # =========================
    # Create Datasets
    # =========================
    print("\nCreating datasets...")
    train_dataset = RockJointDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        polygons_dict=polygons_dict_train,
        patch_size=config['patch_size'],
        normalize_mode=config['normalize_mode'],
        augment=config['augment_train']
    )

    test_dataset = RockJointDataset(
        xyz_array=xyz_array[test_mask],
        label_array=label_array[test_mask],
        polygons_dict=polygons_dict_test,
        patch_size=config['patch_size'],
        normalize_mode=config['normalize_mode'],
        augment=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # =========================
    # Create Model
    # =========================
    print("\nInitializing model...")
    model = PointNet2Segmentation(
        num_classes=config['num_classes'],
        input_channels=config['input_channels']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================
    # Train
    # =========================
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()


if __name__ == '__main__':
    main()
