import torch
import cv2
from pathlib import Path
from src.model import get_model
from src.transforms import get_valid_transforms


def load_model(ckpt_path, cfg):
    """Load trained model + weights."""
    model = get_model(
        backbone_name=cfg.get("backbone", "resnet50"),
        pretrained=False,
        out_dim=7,
        use_transformer=cfg.get("use_transformer", False)
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def infer_single_image(img_path, model, cfg):
    """Run inference on a single image."""
    img = cv2.imread(img_path)[:, :, ::-1]
    transforms = get_valid_transforms(cfg["image_size"])
    transformed = transforms(image=img, bboxes=[], labels=[])

    inp = transformed["image"].unsqueeze(0)  # B=1
    inp = inp.to(cfg["device"])

    with torch.no_grad():
        pred = model(inp)[0].cpu()

    return pred  # 7 values or your bbox representation


def export_to_onnx(model, cfg, out_path="model.onnx"):
    """Export PyTorch model → ONNX."""
    dummy = torch.randn(1, 3, cfg["image_size"], cfg["image_size"])

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
    )

    print(f"Exported ONNX to {out_path}")


def run_inference_folder(folder_path, model, cfg, save_results=False):
    """Run inference on all images inside a folder."""
    folder = Path(folder_path)
    img_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    results = []

    for img_path in img_files:
        pred = infer_single_image(str(img_path), model, cfg)
        results.append({"image": str(img_path), "prediction": pred.tolist()})

        if save_results:
            print(img_path.name, pred.tolist())

    return results