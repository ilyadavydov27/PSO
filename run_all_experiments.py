"""
run_all_experiments.py
======================
Прогоняет все изображения из папки pics/ с водяным знаком pics/ec.png.
Выводит таблицу метрик (PSNR, NCC) аналогично таблицам из статьи.

Запуск:
    cd c:\\Users\\Илья\\PYTHON
    python -m praktos_watermarking.run_all_experiments
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .attacks import default_attacks
from .metrics import ncc, nsr, psnr
from .pso_watermarking import EmbedConfig, PSOParams, embed_watermark, extract_watermark


def _load_gray_u8(path: Path, size=None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, resample=Image.BICUBIC)
    return np.array(im, dtype=np.uint8)


def run_one(cover_path: Path, wm_path: Path, mode: str = "at-pso") -> None:
    cover = _load_gray_u8(cover_path)
    watermark = _load_gray_u8(wm_path)
    wm01 = (watermark > 127).astype(np.uint8)

    attacks = default_attacks()
    use_arnold = mode == "at-pso"

    cfg = EmbedConfig(
        wavelet="haar",
        use_arnold=use_arnold,
        arnold_iters=10,
        pso=PSOParams(
            particles=5,
            iterations=5,
            alpha_min=1.0,
            alpha_max=100.0,
        ),
        attacks_for_fitness=None,
    )

    cw_u8, artifacts = embed_watermark(cover, watermark, config=cfg, attack_fns=attacks)

    img_psnr = psnr(cover, cw_u8)

    # Извлекаем без атаки (проверка "чистого" встраивания)
    w_hat_clean = extract_watermark(cw_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)
    ncc_clean = ncc(wm01.astype(np.float64), w_hat_clean.astype(np.float64))

    cover_name = cover_path.stem
    print(f"\n{'='*70}")
    print(f"Изображение: {cover_name:<12} | Режим: {mode.upper():<6} | alpha={artifacts.alpha:.2f}")
    print(f"PSNR(cover, watermarked) = {img_psnr:.4f} dB")
    print(f"NCC(wm_orig, wm_extracted, no_attack) = {ncc_clean:.6f}")
    print(f"{'-'*70}")
    print(f"{'Атака':<16} | {'PSNR(attacked)':>14} | {'NCC':>9} | {'NSR':>9}")
    print(f"{'-'*70}")

    cw_pil = Image.fromarray(cw_u8, mode="L")
    for key, atk in attacks.items():
        attacked_pil = atk.fn(cw_pil)
        attacked_u8 = np.array(attacked_pil.convert("L"), dtype=np.uint8)
        w_hat = extract_watermark(
            attacked_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5
        )
        atk_psnr = psnr(cw_u8, attacked_u8)
        atk_ncc = ncc(wm01.astype(np.float64), w_hat.astype(np.float64))
        atk_nsr = nsr(wm01, w_hat)
        print(
            f"{key:<16} | {atk_psnr:>14.4f} | {atk_ncc:>9.6f} | {atk_nsr:>9.6f}"
        )


def main() -> None:
    pics_dir = Path(__file__).parent / "pics"
    wm_path = pics_dir / "ec.png"

    if not wm_path.exists():
        raise FileNotFoundError(f"Водяной знак не найден: {wm_path}")

    # Все изображения кроме водяного знака
    cover_names = sorted(
        p for p in pics_dir.glob("*.png") if p.name != "ec.png"
    )

    if not cover_names:
        raise FileNotFoundError(f"Не найдено покрывающих изображений в {pics_dir}")

    print(f"Водяной знак: {wm_path}")
    print(f"Режим: AT-PSO (Arnold + PSO)")
    print(f"Изображений для обработки: {len(cover_names)}")

    for cover_path in cover_names:
        run_one(cover_path, wm_path, mode="at-pso")

    print(f"\n{'='*70}")
    print("Готово.")


if __name__ == "__main__":
    main()
