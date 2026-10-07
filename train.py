import torch
import Dataset
import random
import os
import model
from tqdm import tqdm
import time
import config


def main():
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    path_check = os.path.join(config.PATH, config.DIR_NAME)
    path_last = os.path.join(path_check,'last.pt')
    path_best = os.path.join(path_check,'best.pt')
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    os.makedirs(config.DIR_NAME, exist_ok=True)
      
    ds_t = Dataset.Dataset(path=config.DATA_PATH, isTrain=True, cut_length=config.cut_length, fft_size=config.fft_size, win_length=config.win_length, hop_length=config.hop_length, window=config.window, center=config.center, pad_mode=config.pad_mode, size_multiplier=config.size_multiplier)
    dl_t = torch.utils.data.DataLoader(ds_t, batch_size=config.batch_t, shuffle=True, drop_last=True, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)

    ds_v = Dataset.Dataset(path=config.DATA_PATH, isTrain=False, cut_length=config.cut_length, fft_size=config.fft_size, win_length=config.win_length, hop_length=config.hop_length, window=config.window, center=config.center, pad_mode=config.pad_mode)
    dl_v = torch.utils.data.DataLoader(ds_v, batch_size=config.batch_v, shuffle=False, drop_last=False, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)

    unet = model.UNet(first_layer_channels=config.first_layer_channels, depth=config.depth, eps=config.log_eps)
    unet = unet.to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.patience)

    best_loss = float("inf")
    epoch_count = 0
    start_epoch = 1
    model_config = {"first_layer_channels": config.first_layer_channels, "depth": config.depth, "log": "log10", "lr": config.lr, "scheduler_factor": config.scheduler_factor}
    stft_config = {"n_fft": config.fft_size, "hop_length": config.hop_length, "win_length": config.win_length, "window": config.window, "center": config.center, "pad_mode": config.pad_mode}
    data_config = {"fs": config.fs, "cut_length": config.cut_length, "size_multiplier": config.size_multiplier, "random_seed": config.random_seed}
    run_config = {"model": model_config, "stft": stft_config, "data": data_config}

    if config.resume_from_checkpoint and os.path.exists(path_last):
        last = torch.load(path_last, map_location=device)
        unet.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler'])
        best_loss = last['best_loss']
        epoch_count = last["epochs_without_improvement"]
        start_epoch = last['epoch'] + 1
        run_config = last['config']

    for epoch in range(start_epoch, config.max_epochs + 1):
        start_time = time.perf_counter()

        mean_loss_t = train(unet, dl_t, optimizer, device)
        mean_loss_v = validate(unet, dl_v, device)
        scheduler.step(mean_loss_v)

        duration = time.perf_counter() - start_time
        loss_diff = best_loss - mean_loss_v

        if loss_diff > 1e-4:
            best_loss = mean_loss_v 
            epoch_count = 0 
            best = {"model": unet.state_dict(), "epoch": epoch, "mean_loss_v": mean_loss_v, "config": run_config}
            torch.save(best, path_best)
        else:
            epoch_count += 1
    
        last = {"model": unet.state_dict(), "epoch": epoch, "mean_loss_v": mean_loss_v, "config": run_config, "scheduler": scheduler.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss,"epochs_without_improvement": epoch_count}
        torch.save(last, path_last)

        print(f"Epoch {epoch}: mean_loss_v: {mean_loss_v:.4f}, mean_loss_t: {mean_loss_t:.4f}, duration: {duration:.1f}s, loss_diff: {loss_diff:.4f}, lr: {optimizer.param_groups[0]['lr']}") 

        if epoch_count >= config.early_stopping:
                print(f"Early-stopping at epoch: {epoch}, mean_loss_v: {best_loss:.4f}, last.pt saved to {path_last}")
                break

        



def loss(input_mask, target_mask, amplitude_mixed):
    loss = torch.nn.L1Loss()
    input = input_mask * amplitude_mixed
    target = target_mask * amplitude_mixed
    return loss(input, target)

def train(model, dl_t, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(dl_t):
        mixed_spec, mask_spec = batch
        mixed_spec, mask_spec = mixed_spec.to(device, non_blocking=True), mask_spec.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=False)
        model_mask = model(mixed_spec)
        batch_loss = loss(model_mask, mask_spec, mixed_spec)
        batch_loss.backward()
        optimizer.step()

        num_samples = mixed_spec.shape[0]
        total_loss += batch_loss.item() * num_samples
        total_samples += num_samples 

    return total_loss / total_samples

def validate(model, dl_v, device):
    model.eval()
    with torch.inference_mode():
        total_loss = 0.0
        total_samples = 0
        for batch in tqdm(dl_v):
            mixed_spec, mask_spec = batch
            mixed_spec, mask_spec = mixed_spec.to(device, non_blocking=True), mask_spec.to(device, non_blocking=True)

            model_mask = model(mixed_spec)
            batch_loss = loss(model_mask, mask_spec, mixed_spec)

            num_samples = mixed_spec.shape[0]
            total_loss += batch_loss.item() * num_samples
            total_samples += num_samples 

    return total_loss / total_samples


if __name__ == "__main__":
    main()
