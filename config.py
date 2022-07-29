import argparse
import os

from click import option

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def get_config():
    parser = argparse.ArgumentParser()

    # Run metadata
    parser.add_argument("--tag", default="default", help="Description of this run")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])

    # Dataset args
    parser.add_argument("--input_size", type=int, default=224, help="input size")
    parser.add_argument("--output_size", type=int, default=64, help="output size")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument(
        "--source_dataset_dir",
        type=str,
        default="/datasets/gazefollow_extended",
        help="directory where the source dataset is located",
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="gazefollow",
        choices=["gazefollow", "videoattentiontarget", "goo"],
    )
    parser.add_argument(
        "--target_dataset_dir",
        type=str,
        default="/datasets/gazefollow_extended",
        help="directory where the target dataset is located",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="gazefollow",
        choices=["gazefollow", "videoattentiontarget", "goo"],
    )
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))

    # Model args
    parser.add_argument("--init_weights", type=str, help="initial weights")
    parser.add_argument("--eval_weights", type=str, help="If set, performs evaluation only")

    # Training args
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=70, help="number of epochs")
    parser.add_argument("--evaluate_every", type=int, default=1, help="evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=1, help="save model every N epochs")
    parser.add_argument("--print_every", type=int, default=10, help="print training stats every N batches")
    parser.add_argument("--no_resume", default=False, action="store_true", help="Resume from a stopped run if exists")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output folder")
    parser.add_argument("--amp", type=str, default=None, help="AMP optimization level")
    parser.add_argument("--freeze_scene", default=False, action="store_true", help="Freeze the scene backbone")
    parser.add_argument("--freeze_face", default=False, action="store_true", help="Freeze the head backbone")
    parser.add_argument("--freeze_depth", default=False, action="store_true", help="Freeze the depth backbone")
    parser.add_argument("--head_da", default=False, action="store_true", help="Do DA on head backbone")
    parser.add_argument("--rgb_depth_da", default=False, action="store_true", help="Do DA on rgb/depth backbone")
    parser.add_argument("--task_loss_amp_factor", type=float, default=1)
    parser.add_argument("--rgb_depth_source_loss_amp_factor", type=float, default=1)
    parser.add_argument("--rgb_depth_target_loss_amp_factor", type=float, default=1)
    parser.add_argument("--adv_loss_amp_factor", type=float, default=1)

    parser.add_argument("--no_wandb", default=False, action="store_true", help="Disables wandb")
    parser.add_argument(
        "--no_save",
        default=False,
        action="store_true",
        help="Do not save checkpoint every {save_every}. Stores last checkpoint only to allow resuming",
    )

    args = parser.parse_args()

    # Update output dir
    args.model_id = f"spatial_depth_late_fusion_{args.source_dataset}_{args.target_dataset}"
    args.output_dir = os.path.join(args.output_dir, args.model_id, args.tag)

    # Reverse resume flag to ease my life
    args.resume = not args.no_resume and args.eval_weights is None
    del args.no_resume

    # Reverse wandb flag
    args.wandb = not args.no_wandb
    del args.no_wandb

    # Reverse save flag
    args.save = not args.no_save
    del args.no_save

    # Check if AMP is set and is available. If not, remove amp flag
    if args.amp and amp is None:
        args.amp = None

    # Print configuration
    print(vars(args))
    print()

    return args
