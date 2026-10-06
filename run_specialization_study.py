import sys
sys.path.insert(0, 'c:/Users/Илья/PYTHON')

import numpy as np
from PIL import Image
from pathlib import Path
import collections
import csv

from praktos_watermarking.attacks import default_attacks
from praktos_watermarking.metrics import psnr, ncc
from praktos_watermarking.pso_watermarking import embed_watermark, extract_watermark, EmbedConfig, PSOParams

def main():
    pics_dir = Path("c:/Users/Илья/PYTHON/praktos_watermarking/pics")
    wm_path = pics_dir / "ec.png"
    
    watermark = np.array(Image.open(wm_path).convert("L"), dtype=np.uint8)
    wm01 = (watermark > 127).astype(np.float64)

    images = ["pepper.png", "goldhill.png", "lena512.png", "elaine.png"]
    all_attacks = default_attacks()
    attack_names = list(all_attacks.keys())
    
    runs = 10 
    
    out_md = open('c:/Users/Илья/PYTHON/praktos_watermarking/specialization_results.md', 'w', encoding='utf-8')
    out_csv_path = 'c:/Users/Илья/PYTHON/praktos_watermarking/specialization_results.csv'
    
    def log(msg=""):
        out_md.write(msg + "\n")
        print(msg)

    log("## Исследование специализации PSO (Single-Attack Specialization)")
    log("В данном эксперименте в целевую функцию (ЦФ) PSO включается только ОДНА атака.")
    log(f"Для каждого сценария проводится по {runs} прогонов на 4 изображениях.\n")
    
    # Готовим CSV
    csv_file = open(out_csv_path, 'w', encoding='utf-8-sig', newline='')
    csv_writer = csv.writer(csv_file, delimiter=';')
    header = ["ЦФ (Атака)", "Изображение", "PSNR (clean)", "NCC (clean)"] + [f"NCC ({atk})" for atk in attack_names]
    csv_writer.writerow(header)

    # Итерируемся по каждой атаке, которая будет "учителем" в ЦФ
    for teacher_name in attack_names:
        log(f"### Сценарий: ЦФ оптимизирована под {teacher_name}")
        teacher_attacks = {teacher_name: all_attacks[teacher_name]}
        
        md_header = "| Изображение | PSNR (clean) | NCC (clean) | " + " | ".join([f"NCC ({atk})" for atk in attack_names]) + " |"
        md_sep = "|---|---|---| " + " | ".join(["---" for _ in attack_names]) + " |"
        log(md_header)
        log(md_sep)
        
        cfg = EmbedConfig(
            wavelet="haar",
            use_arnold=True,
            arnold_iters=10,
            pso=PSOParams(particles=5, iterations=8, alpha_min=1.0, alpha_max=300.0),
            attacks_for_fitness=teacher_attacks
        )
        
        for img_name in images:
            cover = np.array(Image.open(pics_dir / img_name).convert("L"), dtype=np.uint8)
            
            run_data = collections.defaultdict(list)
            for r_idx in range(runs):
                cw_u8, artifacts = embed_watermark(cover, watermark, config=cfg, attack_fns=teacher_attacks)
                
                run_data["PSNR_clean"].append(psnr(cover, cw_u8))
                w_hat_clean = extract_watermark(cw_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)
                run_data["NCC_clean"].append(ncc(wm01, w_hat_clean.astype(np.float64)))
                
                cw_pil = Image.fromarray(cw_u8, mode="L")
                for atk_name, atk in all_attacks.items():
                    attacked_pil = atk.fn(cw_pil)
                    attacked_u8 = np.array(attacked_pil.convert("L"), dtype=np.uint8)
                    w_hat = extract_watermark(attacked_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)
                    run_data[f"NCC_{atk_name}"].append(ncc(wm01, w_hat.astype(np.float64)))
            
            m_psnr = np.mean(run_data["PSNR_clean"])
            s_psnr = np.std(run_data["PSNR_clean"])
            m_ncc_c = np.mean(run_data["NCC_clean"])
            s_ncc_c = np.std(run_data["NCC_clean"])
            
            # Строка для MD
            md_row = [f"| {img_name} | {m_psnr:.2f}±{s_psnr:.2f} | {m_ncc_c:.4f}±{s_ncc_c:.4f} |"]
            # Строка для CSV
            csv_row = [teacher_name, img_name, f"{m_psnr:.2f}±{s_psnr:.2f}", f"{m_ncc_c:.4f}±{s_ncc_c:.4f}"]
            
            for atk_name in attack_names:
                m_atk = np.mean(run_data[f"NCC_{atk_name}"])
                s_atk = np.std(run_data[f"NCC_{atk_name}"])
                md_row.append(f"{m_atk:.4f}±{s_atk:.4f}")
                csv_row.append(f"{m_atk:.4f}±{s_atk:.4f}")
                
            log(" ".join(md_row) + " |")
            csv_writer.writerow(csv_row)
            out_md.flush()
            csv_file.flush()
        log("\n")
        
    out_md.close()
    csv_file.close()
    print(f"Specialization study complete! Files: {out_csv_path}, specialization_results.md")

if __name__ == "__main__":
    main()
