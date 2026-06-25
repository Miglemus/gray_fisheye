from gray.imports import *

import warnings
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from piq import LPIPS
from gray.utils import masked_psnr, masked_ssim


@dataclass
class MetricsCLI:
    model_paths: Annotated[List[str], arg(aliases=["-m"])]
    batch_size: int = 1


class ImagePairDataset(Dataset):
    def __init__(self, renders_dir: Path, gt_dir: Path):
        self.renders_dir = renders_dir
        self.gt_dir = gt_dir
        self.fnames = sorted(
            fname for fname in os.listdir(renders_dir) if (gt_dir / fname).exists()
        )

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        render = read_image(str(self.renders_dir / fname), ImageReadMode.RGB).float() / 255.0
        gt = read_image(str(self.gt_dir / fname), ImageReadMode.RGB).float() / 255.0
        return render, gt, fname


def discover_metric_dirs(method_dir: Path):
    mode_dirs = [
        (entry.name, entry)
        for entry in sorted(method_dir.iterdir())
        if entry.is_dir() and (entry / "renders").exists()
    ]
    if mode_dirs:
        return mode_dirs
    if (method_dir / "renders").exists():
        return [(None, method_dir)]
    return []


def load_mask(method_dir: Path):
    mask_path = method_dir / "valid_mask.png"
    if not mask_path.exists():
        return None
    return (read_image(str(mask_path), ImageReadMode.GRAY).float()[0].cuda() / 255.0) > 0.5


if __name__ == "__main__":
    # * Parse Config
    cli = tyro.cli(MetricsCLI)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_fn = LPIPS(reduction="none").cuda()

    avg_metrics = {}
    per_image_metrics = {}

    for model_path in cli.model_paths:
        print("Scene:", model_path)
        avg_metrics[model_path] = {}
        per_image_metrics[model_path] = {}

        test_dir = Path(model_path) / "test"
        per_mode_avg_metrics = {}
        per_mode_image_metrics = {}

        for method in sorted(os.listdir(test_dir)):
            print("Iterations:", method)
            method_dir = test_dir / method
            metric_dirs = discover_metric_dirs(method_dir)
            for mode, mode_dir in metric_dirs:
                if mode is not None:
                    print("Mode:", mode)
                if not (mode_dir / "gt").exists():
                    print("  Skipping metrics: no gt folder found")
                    continue
                dataset = ImagePairDataset(mode_dir / "renders", mode_dir / "gt")
                if len(dataset) == 0:
                    print("  Skipping metrics: no matching render/gt image pairs found")
                    continue
                loader = DataLoader(dataset, batch_size=cli.batch_size, num_workers=4, pin_memory=True)
                valid_mask = load_mask(mode_dir)

                ssim_scores = []
                psnr_scores = []
                lpips_scores = []
                image_names = []

                for renders_batch, gts_batch, fnames_batch in tqdm(loader, desc="Metric evaluation progress"):
                    renders_batch = renders_batch.cuda()
                    gts_batch = gts_batch.cuda()

                    if valid_mask is None:
                        ssim_scores.extend(
                            ssim(renders_batch, gts_batch, downsample=False, reduction="none").tolist()
                        )
                        psnr_scores.extend(psnr(renders_batch, gts_batch, reduction="none").tolist())
                    else:
                        for idx in range(renders_batch.shape[0]):
                            ssim_scores.append(
                                masked_ssim(renders_batch[idx], gts_batch[idx], valid_mask).item()
                            )
                            psnr_scores.append(
                                masked_psnr(renders_batch[idx], gts_batch[idx], valid_mask).item()
                            )
                    lpips_scores.extend(lpips_fn(renders_batch, gts_batch).tolist())
                    image_names.extend(fnames_batch)

                ssim_mean = torch.tensor(ssim_scores).mean().item()
                psnr_mean = torch.tensor(psnr_scores).mean().item()
                lpips_mean = torch.tensor(lpips_scores).mean().item()

                print("  SSIM : {:>12.7f}".format(ssim_mean))
                print("  PSNR : {:>12.7f}".format(psnr_mean))
                print("  LPIPS: {:>12.7f}".format(lpips_mean))

                summary = {"SSIM": ssim_mean, "PSNR": psnr_mean, "LPIPS": lpips_mean}
                per_image = {
                    "SSIM": {name: val for name, val in zip(image_names, ssim_scores)},
                    "PSNR": {name: val for name, val in zip(image_names, psnr_scores)},
                    "LPIPS": {name: val for name, val in zip(image_names, lpips_scores)},
                }

                if mode is None:
                    avg_metrics[model_path][method] = summary
                    per_image_metrics[model_path][method] = per_image
                else:
                    avg_metrics[model_path].setdefault(method, {})[mode] = summary
                    per_image_metrics[model_path].setdefault(method, {})[mode] = per_image
                    per_mode_avg_metrics.setdefault(mode, {})[method] = summary
                    per_mode_image_metrics.setdefault(mode, {})[method] = per_image

        # * Save results
        with open(model_path + "/results.json", "w") as fp:
            json.dump(avg_metrics[model_path], fp, indent=True)
        with open(model_path + "/per_view.json", "w") as fp:
            json.dump(per_image_metrics[model_path], fp, indent=True)
        for mode, results in per_mode_avg_metrics.items():
            with open(model_path + f"/results_{mode}.json", "w") as fp:
                json.dump(results, fp, indent=True)
        for mode, results in per_mode_image_metrics.items():
            with open(model_path + f"/per_view_{mode}.json", "w") as fp:
                json.dump(results, fp, indent=True)
