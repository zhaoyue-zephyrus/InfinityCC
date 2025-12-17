import glob
import json

import cv2
import torch
import torch.distributed as dist
from tqdm import tqdm
import traceback
torch._dynamo.config.cache_size_limit = 64

from run_infinity import *


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def create_npz_from_sample_folder_v2(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for img_path in tqdm(glob.glob(sample_dir + '/*.png'), desc="Building .npz file from samples"):
        sample_pil = Image.open(img_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--fid_max_examples', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--num_per_class', type=int, default=50)
    parser.add_argument('--bs', type=int, default=64)
    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # parse top_k and top_p like cfg - can be single value or comma-separated list
    args.top_k = list(map(int, args.top_k.split(',')))
    if len(args.top_k) == 1:
        args.top_k = args.top_k[0]
    
    args.top_p = list(map(float, args.top_p.split(',')))
    if len(args.top_p) == 1:
        args.top_p = args.top_p[0]
        
    if args.out_dir:
        out_dir = args.out_dir
    else:
        raise ValueError('out_dir is required')
    print(f'save to {out_dir}')

    # setup ddp
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = args.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    # inference
    if osp.exists(out_dir):
        # shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(__file__, osp.join(out_dir, osp.basename(__file__)))
    if not osp.exists(osp.join(out_dir, 'pred')):
        os.makedirs(osp.join(out_dir, 'pred'), exist_ok=True)

    if args.schedule_mode == "original":
        h_div_w_template = 0.
    else:
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (t, h, w) in scale_schedule]
    
    class_labels = torch.arange(1000).repeat_interleave(args.num_per_class)
    # class_labels = torch.tensor([0,0,0,0,], dtype=torch.int64)
    bs = args.bs
    for i in tqdm(range(rank, math.ceil(class_labels.shape[0] / bs), world_size), desc="Generating images", disable=rank != 0):
        img_list = gen_imgs_class_conditioned(
            infinity,
            vae,
            class_labels[i*bs:(i+1)*bs].cuda(),
            g_seed=seed + i * world_size + rank,
            gt_leak=0,
            gt_ls_Bl=[],
            tau_list=args.tau,
            cfg_sc=3,
            cfg_list=args.cfg,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        for j in range(bs):
            if i * bs + j < class_labels.shape[0]:
                cv2.imwrite(osp.join(out_dir, 'pred', f'pred_{i*bs+j:06d}_class_{class_labels[i*bs+j].item():04d}.png'), img_list[j].cpu().numpy())

    torch.distributed.barrier()
    if rank == 0:
        create_npz_from_sample_folder_v2(osp.join(out_dir, 'pred'), num=class_labels.shape[0])

    exit()