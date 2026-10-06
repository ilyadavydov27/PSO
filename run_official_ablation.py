import sys
sys.path.insert(0, 'c:/Users/Илья/PYTHON')

import numpy as np
from PIL import Image
from pathlib import Path
import collections

from praktos_watermarking.attacks import default_attacks
from praktos_watermarking.metrics import psnr, ncc
from praktos_watermarking.pso_watermarking import embed_watermark, extract_watermark, EmbedConfig, PSOParams

def main():
    pics_dir = Path("c:/Users/Илья/PYTHON/praktos_watermarking/pics")
    wm_path = pics_dir / "ec.png"
    
    watermark = np.array(Image.open(wm_path).convert("L"), dtype=np.uint8)
    wm01 = (watermark > 127).astype(np.float64)

    # 4 изображения строго по заданию
    images = ["pepper.png", "goldhill.png", "lena512.png", "elaine.png"]
    
    all_attacks = default_attacks()
    attack_names = list(all_attacks.keys()) # 7 attacks: blur, scale, hist, rot, jpeg, crop, salt
    
    runs = 10 # Ровно 10 раз на каждом изображении
    
    # Полный цикл от 7 до 0 атак для наглядной динамики
    steps_r = [7, 6, 5, 4, 3, 2, 1, 0]
    
    out_file = open('c:/Users/Илья/PYTHON/praktos_watermarking/ablation_final.md', 'w', encoding='utf-8')
    def log(msg=""):
        out_file.write(msg + "\n")
        print(msg)

    log("## Финальное исследование целевой функции (Ablation Study)")
    log("Эксперимент проведен строго по заданию: 4 изображения, 10 прогонов для учета случайности PSO.\n")
    
    for k in steps_r:
        fitness_attack_names = attack_names[:k] # Берем первые k атак
        fitness_attacks = {name: all_attacks[name] for name in fitness_attack_names}
        
        log(f"### Сценарий: {k} атак в целевой функции")
        if k > 0:
            log(f"**Атаки в фитнесе:** {', '.join(fitness_attack_names)}\n")
        else:
            log("**Атаки в фитнесе:** Нет (Алгоритм слеп)\n")
            
        header = "| Изображение | PSNR (clean) | NCC (clean) | " + " | ".join([f"NCC ({atk})" for atk in attack_names]) + " |"
        sep = "|---|---|---| " + " | ".join(["---" for _ in attack_names]) + " |"
        log(header)
        log(sep)

        cfg = EmbedConfig(
            wavelet="haar",
            use_arnold=True,
            arnold_iters=10,
            pso=PSOParams(particles=5, iterations=5, alpha_min=1.0, alpha_max=300.0),
            attacks_for_fitness=fitness_attacks if k > 0 else {}
        )
        
        for img_name in images:
            cover = np.array(Image.open(pics_dir / img_name).convert("L"), dtype=np.uint8)
            
            run_data = collections.defaultdict(list)
            for r_idx in range(runs):
                cw_u8, artifacts = embed_watermark(cover, watermark, config=cfg, attack_fns=fitness_attacks if k > 0 else {})
                
                # Оценивает каждый раз PSNR без атак:
                run_data["PSNR_clean"].append(psnr(cover, cw_u8))
                
                # и NCC без атак:
                w_hat_clean = extract_watermark(cw_u8, artifacts, watermark_shape=wm01.shape, threshold=0.5)
                run_data["NCC_clean"].append(ncc(wm01, w_hat_clean.astype(np.float64)))
                
                # и со всеми атаками из статьи (NSR / PSNR_attacked считать не надо):
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
            
            row = [f"| {img_name} | {m_psnr:.2f}±{s_psnr:.2f} dB | {m_ncc_c:.4f}±{s_ncc_c:.4f} |"]
            for atk_name in attack_names:
                m_atk = np.mean(run_data[f"NCC_{atk_name}"])
                s_atk = np.std(run_data[f"NCC_{atk_name}"])
                row.append(f"{m_atk:.4f}±{s_atk:.4f}")
                
            log(" ".join(row) + " |")
            out_file.flush()
        log("\n")
    out_file.close()
    
    print("- Конвертация в CSV...")
    import csv
    with open('c:/Users/Илья/PYTHON/praktos_watermarking/ablation_final.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('c:/Users/Илья/PYTHON/praktos_watermarking/ablation_final_results.csv', 'w', encoding='utf-8-sig', newline='') as out:
        writer = csv.writer(out, delimiter=';')
        for line in lines:
            line = line.strip()
            if line.startswith('### '):
                writer.writerow([])
                writer.writerow([line.replace('### ', '')])
            elif line.startswith('**Атаки'):
                writer.writerow([line.replace('**', '')])
                writer.writerow([])
            elif line.startswith('|') and not line.startswith('|---'):
                cells = [cell.strip().replace('**', '') for cell in line.strip('|').split('|')]
                if "Изображение" in cells[0]:
                    writer.writerow(cells)
                else:
                    if len(cells) >= 3:
                        row = [cells[0], cells[1], cells[2]] + (cells[3].split() if len(cells) > 3 else [])
                        writer.writerow(row)
                    else:
                        writer.writerow(cells)
    print("- Полностью завершено!")

if __name__ == '__main__':
    main()
