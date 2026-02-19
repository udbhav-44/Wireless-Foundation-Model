
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR

# Setup path to import from root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from torch import amp as torch_amp
except Exception:
    torch_amp = None

# Imports
from lwm.input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning
from lwm_axial.torch_pipeline_axial import LWMWithPrepatchAxial

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Fallback to non-distributed
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def default_num_workers():
    return 8

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Pretraining for Axial LWM.")
    # ... Copy args from original ...
    parser.add_argument("--scenarios", nargs="+", default=["O1_3p5_v1", "O1_3p5_v2", "Boston5G_3p5", "asu_campus1", "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston", "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_8_dallas", "city_9_sanfrancisco", "city_10_austin", "city_13_columbus", "city_17_seattle"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512, help="Per-GPU batch size")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tb-logdir", type=str, default="runs/lwm_axial_ddp")
    parser.add_argument("--scheduler-step-per-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-folder", type=str, default="/home/audbhav22/foundation_model/Dataset")
    parser.add_argument("--channels-cache", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="lwm_axial/model_weights_axial_ddp.pth")
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()

def load_channels_ri(scenarios, dataset_folder=None, cache_path=None, rank=0):
    # Only Rank 0 prints details
    if cache_path:
        cache_path = os.path.expanduser(cache_path)
        if os.path.exists(cache_path) or (not cache_path.endswith(".npy") and os.path.exists(cache_path + ".npy")):
            if rank == 0: print(f"Loading cached channels from {cache_path}")
            return np.load(cache_path if os.path.exists(cache_path) else cache_path + ".npy")
    
    if rank == 0: print("Loading data from scenarios...")
    data = []
    for name in scenarios:
        deepmimo_data = DeepMIMO_data_gen(name, dataset_folder=dataset_folder)
        cleaned = deepmimo_data_cleaning(deepmimo_data)
        data.append(cleaned)
    channels = np.vstack(data)
    real = channels.real.astype(np.float32)
    imag = channels.imag.astype(np.float32)
    channels_ri = np.stack([real, imag], axis=1)
    
    if cache_path and rank == 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, channels_ri)
        print(f"Cached channels to {cache_path}")
        
    return channels_ri

def split_data(dataset, train_ratio, val_ratio, seed=0):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - val_size - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)

def train_epoch(model, dataloader, optimizer, scheduler, device, amp, scaler, log_interval, log_fn, writer, epoch_idx, rank, grad_clip, scheduler_step_per_batch, world_size):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()
    
    # Wait for all processes to start epoch
    if dist.is_initialized():
        dataloader.sampler.set_epoch(epoch_idx)

    end = time.perf_counter()
    log_loss = 0.0
    log_data_time = 0.0
    log_step_time = 0.0
    
    for step, (channels,) in enumerate(dataloader):
        data_time = time.perf_counter() - end
        log_data_time += data_time
        
        channels = channels.to(device, non_blocking=True)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp):
            logits_lm, masked_tokens, _ = model(channels)
            loss = criterion(logits_lm, masked_tokens) / torch.var(masked_tokens)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler and scheduler_step_per_batch:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        log_step_time += step_time

        # Reduce loss for logging
        loss_val = loss.item()
        if dist.is_initialized():
            loss_tensor = torch.tensor(loss_val, device=device)
            dist.all_reduce(loss_tensor)
            loss_val = loss_tensor.item() / world_size

        running_loss += loss_val
        log_loss += loss_val
        
        if rank == 0 and log_interval and (step + 1) % log_interval == 0:
            avg_loss = log_loss / log_interval
            avg_data = log_data_time / log_interval
            avg_step = log_step_time / log_interval
            throughput = (args.batch_size * world_size) / avg_step # Samples per second based on step time (processing)
            # OR total throughput including data wait:
            # throughput = (args.batch_size * world_size * log_interval) / (time.perf_counter() - (end - data_time)) 
            # Original uses pure step time or total time? 
            # Original: throughput = log_samples / log_step_time. (Pure processing speed). 
            # I will stick to pure Step Time for throughput to show GPU speed, but Total Time is also useful.
            # Let's use the interval time for "real world" throughput.
            
            # Re-calculating end-to-end throughput for honesty
            # But let's match original script style:
            # f"data {avg_data * 1000:.1f} ms | step {avg_step * 1000:.1f} ms"
            
            log_fn(
                f"  step {step + 1:>5}/{len(dataloader)} | "
                f"loss {avg_loss:.4f} | "
                f"data {avg_data * 1000:.1f} ms | "
                f"step {avg_step * 1000:.1f} ms | "
                f"{throughput:.1f} samples/s"
            )
            
            if writer:
                global_step = epoch_idx * len(dataloader) + step + 1
                writer.add_scalar("train/loss_step", avg_loss, global_step)
                writer.add_scalar("train/data_time_ms", avg_data * 1000, global_step)
                writer.add_scalar("train/step_time_ms", avg_step * 1000, global_step)
                writer.add_scalar("train/throughput", throughput, global_step)

            log_loss = 0.0
            log_data_time = 0.0
            log_step_time = 0.0
            
        end = time.perf_counter()

    return running_loss / len(dataloader)

def main():
    global args
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load Data (All ranks load, mostly likely mapped to memory)
    # Ideally Rank 0 loads and scatters, but for simplicity each loads.
    channels_ri = load_channels_ri(args.scenarios, args.dataset_folder, args.channels_cache, rank)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(channels_ri))
    train_data, val_data, test_data = split_data(dataset, args.train_ratio, args.val_ratio, args.seed)
    
    # Distributed Samplers
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False) if len(val_data) > 0 else None
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory) if val_sampler else None
    
    # Model
    model = LWMWithPrepatchAxial(gen_raw=False, snr_db=args.snr_db).to(device)
    if args.torch_compile: # Compile before DDP usually better
        model = torch.compile(model, mode=args.compile_mode)
        
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    
    # Logging
    log_fp = None
    writer = None
    if rank == 0:
        base_path = args.save_path or "lwm_ddp"
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        log_fp = open(base_path + ".log", "a")
        if args.tensorboard and SummaryWriter:
            writer = SummaryWriter(log_dir=args.tb_logdir)
            
    def log(msg):
        if rank == 0:
            print(msg)
            if log_fp: log_fp.write(str(msg) + "\n"); log_fp.flush()

    log(f"Starting DDP Training: World Size {world_size}, Batch Size {args.batch_size} (Global {args.batch_size*world_size})")

    for epoch in range(args.epochs):
        log(f"Epoch {epoch+1}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.amp, scaler, args.log_interval, log, writer, epoch, rank, args.grad_clip, args.scheduler_step_per_batch, world_size)
        log(f"Train Loss: {avg_loss:.4f}")
        
        if not args.scheduler_step_per_batch:
            scheduler.step()
            
        if rank == 0 and args.save_path:
            # Save only on Rank 0
            torch.save(model.module.state_dict(), args.save_path)
            
    cleanup_ddp()

if __name__ == "__main__":
    main()
