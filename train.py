import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import time
from tqdm import tqdm
from model import ChessNet
from utils import DEVICE
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import lmdb
import pickle

# Updated ChessDataset using LMDB
class ChessDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = f'{idx:08}'.encode('ascii')
        with self.env.begin(write=False) as txn:
            data = txn.get(idx)
            if data is None:
                raise IndexError(f"Index {idx} out of bounds")
            sample = pickle.loads(data)
            board_tensor = sample['board_tensor']
            move_index = sample['move_index']
            value_target = sample['value_target']
            return (
                torch.tensor(board_tensor, dtype=torch.float32),
                torch.tensor(move_index, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32)
            )

    def __del__(self):
        self.env.close()

def train_model(dataset_path, num_epochs=10, batch_size=64, learning_rate=0.001,
                patience=5, min_delta=0.001, num_workers=4, run_name='chess_training'):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    dataset = ChessDataset(dataset_path)

    # Split dataset into training and validation sets
    val_split = 0.1  # 10% for validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_value_loss = 0
        total_policy_loss = 0
        total_loss = 0

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            # Check for pause signal
            if os.path.exists('pause'):
                print('Training paused. Remove the "pause" file to continue.')
                while os.path.exists('pause'):
                    time.sleep(10)
                print('Training resumed.')

            board_tensors, move_indices, value_targets = batch
            board_tensors = board_tensors.to(DEVICE, non_blocking=True)
            move_indices = move_indices.to(DEVICE, non_blocking=True)
            value_targets = value_targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_values, policy_logits = model(board_tensors)

                # Value loss
                v_loss = value_loss_fn(predicted_values, value_targets)

                # Policy loss
                p_loss = policy_loss_fn(policy_logits, move_indices)

                loss = v_loss + p_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_value_loss += v_loss.item()
            total_policy_loss += p_loss.item()
            total_loss += loss.item()

            # Global step for TensorBoard
            global_step = epoch * len(train_dataloader) + batch_idx

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
            writer.add_scalar('Loss/Train_Value', v_loss.item(), global_step)
            writer.add_scalar('Loss/Train_Policy', p_loss.item(), global_step)

            pbar.set_postfix({'Value Loss': v_loss.item(), 'Policy Loss': p_loss.item()})

        avg_value_loss = total_value_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_total_loss = total_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        val_value_loss = 0
        val_policy_loss = 0
        val_total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                board_tensors, move_indices, value_targets = batch
                board_tensors = board_tensors.to(DEVICE, non_blocking=True)
                move_indices = move_indices.to(DEVICE, non_blocking=True)
                value_targets = value_targets.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast():
                    predicted_values, policy_logits = model(board_tensors)

                    v_loss = value_loss_fn(predicted_values, value_targets)
                    p_loss = policy_loss_fn(policy_logits, move_indices)
                    loss = v_loss + p_loss

                val_value_loss += v_loss.item()
                val_policy_loss += p_loss.item()
                val_total_loss += loss.item()

        avg_val_value_loss = val_value_loss / len(val_dataloader)
        avg_val_policy_loss = val_policy_loss / len(val_dataloader)
        avg_val_total_loss = val_total_loss / len(val_dataloader)

        # Log validation losses to TensorBoard
        writer.add_scalar('Loss/Epoch_Train_Total', avg_total_loss, epoch)
        writer.add_scalar('Loss/Epoch_Train_Value', avg_value_loss, epoch)
        writer.add_scalar('Loss/Epoch_Train_Policy', avg_policy_loss, epoch)
        writer.add_scalar('Loss/Epoch_Val_Total', avg_val_total_loss, epoch)
        writer.add_scalar('Loss/Epoch_Val_Value', avg_val_value_loss, epoch)
        writer.add_scalar('Loss/Epoch_Val_Policy', avg_val_policy_loss, epoch)

        print(f'Epoch {epoch+1}, '
              f'Train Loss: {avg_total_loss:.4f}, '
              f'Train Value Loss: {avg_value_loss:.4f}, '
              f'Train Policy Loss: {avg_policy_loss:.4f}, '
              f'Val Loss: {avg_val_total_loss:.4f}, '
              f'Val Value Loss: {avg_val_value_loss:.4f}, '
              f'Val Policy Loss: {avg_val_policy_loss:.4f}')

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_total_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Early stopping
        if avg_val_total_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_total_loss
            epochs_no_improve = 0
            # Save the best model
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_filename = f'chess_model_epoch_{epoch+1}_{timestamp}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_val_total_loss,
            }, model_filename)
            print(f'Validation loss improved. Model saved as {model_filename}')
        else:
            epochs_no_improve += 1
            print(f'No improvement in validation loss for {epochs_no_improve} epochs.')

        if epochs_no_improve >= patience:
            print('Early stopping.')
            break

    # Close the TensorBoard writer
    writer.close()
    print('Training completed.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to the LMDB dataset file', required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--run_name', type=str, default='chess_training', help='Name for the TensorBoard run')
    args = parser.parse_args()

    train_model(
        args.dataset_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        run_name=args.run_name
    )
