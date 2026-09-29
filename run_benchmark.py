"""
run_benchmark.py — Enhanced OCR Benchmark Evaluation

Evaluasi model OCR terhadap Ground Truth menggunakan metrik:
  - CER (Character Error Rate) — standar dan normalisasi
  - WER (Word Error Rate) 
  - Keyword Accuracy — persentase keyword pseudocode yang benar terdeteksi
  - Inference time tracking

Cara run:
  python run_benchmark.py --exp 1          # Evaluasi exp1
  python run_benchmark.py --exp 2          # Evaluasi exp2
  python run_benchmark.py --exp 3          # Evaluasi exp3
  python run_benchmark.py --exp 1 2 3      # Bandingkan semua exp
"""

import os
import glob
import time
import argparse
import sys
from pathlib import Path
import jiwer
from datetime import datetime


# Keyword pseudocode untuk evaluasi keyword accuracy
PSEUDOCODE_KEYWORDS = {
    "function", "endfunction", "procedure", "endprocedure",
    "kamus", "algoritma", "return", "if", "then", "else",
    "endif", "while", "endwhile", "for", "endfor",
    "repeat", "until", "integer", "real", "string", 
    "boolean", "input", "output", "array", "true", "false",
    "not", "and", "or", "mod", "div", "endprogram", "program",
}


def calculate_cer(reference, hypothesis, normalize=False):
    if normalize:
        # Lowercase and replace multiple spaces/newlines with a single space
        ref = " ".join(reference.lower().split())
        hyp = " ".join(hypothesis.lower().split())
    else:
        ref = reference.strip()
        hyp = hypothesis.strip()
        
    if not ref:
        return 1.0 if hyp else 0.0
        
    return jiwer.cer(ref, hyp)


def calculate_wer(reference, hypothesis, normalize=False):
    """Calculate Word Error Rate."""
    if normalize:
        ref = " ".join(reference.lower().split())
        hyp = " ".join(hypothesis.lower().split())
    else:
        ref = reference.strip()
        hyp = hypothesis.strip()
    
    if not ref:
        return 1.0 if hyp else 0.0
    
    return jiwer.wer(ref, hyp)


def calculate_keyword_accuracy(reference, hypothesis):
    """
    Calculate percentage of pseudocode keywords correctly detected.
    Returns (accuracy, found_keywords, total_keywords).
    """
    ref_words = set(w.lower() for w in reference.split())
    hyp_words = set(w.lower() for w in hypothesis.split())
    
    # Keywords present in reference
    ref_keywords = ref_words & PSEUDOCODE_KEYWORDS
    
    if not ref_keywords:
        return 1.0, 0, 0
    
    # Keywords correctly found in hypothesis
    found_keywords = ref_keywords & hyp_words
    
    accuracy = len(found_keywords) / len(ref_keywords)
    return accuracy, len(found_keywords), len(ref_keywords)


def get_clean_text(path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[WARNING] Gagal membaca {path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Enhanced OCR Benchmark Evaluation.")
    parser.add_argument("--gt_dir", type=str, default=str(Path(__file__).parent / "ground_truth"), 
                        help="Path to Ground Truth md files")
    parser.add_argument("--exp", nargs='+', type=str, default=["1"], 
                        help="Eksperimen yang dievaluasi (bisa multiple: --exp 1 2 3)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of ground truth files processed (0 for all)")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("RUNNING ENHANCED BENCHMARK EVALUATION")
    print(f"Metrics: CER, WER, Keyword Accuracy")
    print("=" * 80)
    print(f"Ground Truth Directory: {gt_dir}")
    print(f"Experiments: {', '.join(args.exp)}")
    if args.limit > 0:
        print(f"Limit: First {args.limit} ground truth files")

    # Ensure GT exists
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Folder Ground Truth kosong. Buat GT dulu.")
        return

    gt_files = sorted(list(gt_dir.glob("*.md")))
    
    # Filter out _penjelasan.md files since they don't have matching OCR outputs
    gt_files = [f for f in gt_files if not f.name.endswith("_penjelasan.md")]

    if not gt_files:
        print(f"\n[WARNING] Tidak ada file Ground Truth (.md) di {gt_dir}")
        return

    if args.limit > 0:
        gt_files = gt_files[:args.limit]

    print(f"Ditemukan {len(gt_files)} file Ground Truth yang akan dievaluasi.")

    # Find result folders for each experiment
    result_folders = []  # List of (model_name, exp_name, path)
    
    for exp in args.exp:
        for model_group in ["paddleocr", "trocr"]:
            res_dir = base_dir / model_group / "results" / f"exp{exp}"
            if res_dir.exists() and res_dir.is_dir():
                result_folders.append((model_group, f"exp{exp}", res_dir))

    if not result_folders:
        print("[WARNING] Tidak ada folder hasil yang ditemukan.")
        return

    print("Folder model yang terdeteksi:")
    for model, exp, path in result_folders:
        print(f"  - {model} / {exp}")

    # Process metrics
    evaluation_results = {}
    model_averages = {}
    
    for model, exp, _ in result_folders:
        key = f"{model}/{exp}"
        model_averages[key] = {
            "cer_raw_sum": 0.0, "cer_norm_sum": 0.0,
            "wer_raw_sum": 0.0, "wer_norm_sum": 0.0,
            "kw_acc_sum": 0.0, "kw_found": 0, "kw_total": 0,
            "count": 0
        }

    for gt_path in gt_files:
        img_base = gt_path.stem
        gt_text = get_clean_text(gt_path)
        if gt_text is None:
            continue
            
        evaluation_results[img_base] = {"gt_len": len(gt_text), "models": {}}

        for model, exp, rf in result_folders:
            key = f"{model}/{exp}"
            
            model_out_path_txt = rf / f"{img_base}.txt"
            model_out_path_md = rf / f"{img_base}.md"
            
            model_text = None
            if model_out_path_txt.exists():
                model_text = get_clean_text(model_out_path_txt)
            elif model_out_path_md.exists():
                model_text = get_clean_text(model_out_path_md)

            if model_text is not None:
                cer_raw = calculate_cer(gt_text, model_text, normalize=False)
                cer_norm = calculate_cer(gt_text, model_text, normalize=True)
                wer_raw = calculate_wer(gt_text, model_text, normalize=False)
                wer_norm = calculate_wer(gt_text, model_text, normalize=True)
                kw_acc, kw_found, kw_total = calculate_keyword_accuracy(gt_text, model_text)
                
                evaluation_results[img_base]["models"][key] = {
                    "cer_raw": cer_raw, "cer_norm": cer_norm,
                    "wer_raw": wer_raw, "wer_norm": wer_norm,
                    "kw_acc": kw_acc, "kw_found": kw_found, "kw_total": kw_total,
                    "len": len(model_text)
                }
                
                model_averages[key]["cer_raw_sum"] += cer_raw
                model_averages[key]["cer_norm_sum"] += cer_norm
                model_averages[key]["wer_raw_sum"] += wer_raw
                model_averages[key]["wer_norm_sum"] += wer_norm
                model_averages[key]["kw_acc_sum"] += kw_acc
                model_averages[key]["kw_found"] += kw_found
                model_averages[key]["kw_total"] += kw_total
                model_averages[key]["count"] += 1
            else:
                evaluation_results[img_base]["models"][key] = None

    # Print summary
    print("\n" + "=" * 100)
    print("RINGKASAN EVALUASI")
    print("=" * 100)
    header = f"{'Model/Exp':<25} | {'N':<4} | {'CER Std':<10} | {'CER Norm':<10} | {'WER Std':<10} | {'WER Norm':<10} | {'KW Acc':<10}"
    print(header)
    print("-" * 100)
    
    for key, data in model_averages.items():
        if data["count"] > 0:
            avg_cer_raw = (data["cer_raw_sum"] / data["count"]) * 100
            avg_cer_norm = (data["cer_norm_sum"] / data["count"]) * 100
            avg_wer_raw = (data["wer_raw_sum"] / data["count"]) * 100
            avg_wer_norm = (data["wer_norm_sum"] / data["count"]) * 100
            avg_kw = (data["kw_acc_sum"] / data["count"]) * 100
            print(f"{key:<25} | {data['count']:<4} | {avg_cer_raw:>8.2f}% | {avg_cer_norm:>8.2f}% | {avg_wer_raw:>8.2f}% | {avg_wer_norm:>8.2f}% | {avg_kw:>8.2f}%")
        else:
            print(f"{key:<25} | 0    | {'N/A':>9} | {'N/A':>9} | {'N/A':>9} | {'N/A':>9} | {'N/A':>9}")
    print("=" * 100)

    # Generate Markdown Report
    exp_label = "_".join(args.exp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = base_dir / "benchmark_results" / f"benchmark_exp{exp_label}_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_md = f"""# Laporan Evaluasi OCR — Experiment {', '.join(args.exp)}

Tanggal: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 1. Ringkasan Akurasi Rata-Rata

| Model / Experiment | Sampel | CER (Standar) | CER (Normalisasi) | WER (Standar) | WER (Normalisasi) | Keyword Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for key, data in model_averages.items():
        if data["count"] > 0:
            avg_cer_raw = (data["cer_raw_sum"] / data["count"]) * 100
            avg_cer_norm = (data["cer_norm_sum"] / data["count"]) * 100
            avg_wer_raw = (data["wer_raw_sum"] / data["count"]) * 100
            avg_wer_norm = (data["wer_norm_sum"] / data["count"]) * 100
            avg_kw = (data["kw_acc_sum"] / data["count"]) * 100
            report_md += f"| **{key}** | {data['count']} | {avg_cer_raw:.2f}% | {avg_cer_norm:.2f}% | {avg_wer_raw:.2f}% | {avg_wer_norm:.2f}% | {avg_kw:.2f}% |\n"
        else:
            report_md += f"| **{key}** | 0 | N/A | N/A | N/A | N/A | N/A |\n"

    report_md += """
\\* *CER Normalisasi: lowercase + pembersihan spasi/line break ganda.*
\\* *Keyword Accuracy: persentase keyword pseudocode (function, kamus, algoritma, dll.) yang terdeteksi benar.*

---

## 📄 2. Detail Evaluasi per Gambar

"""
    for img_name, data in evaluation_results.items():
        report_md += f"### Gambar: `{img_name}` (Ground Truth: {data['gt_len']} karakter)\n\n"
        report_md += "| Model/Exp | Output | CER Std | CER Norm | WER Std | WER Norm | KW Acc |\n"
        report_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for key, m_res in data["models"].items():
            if m_res is not None:
                kw_str = f"{m_res['kw_acc']*100:.0f}% ({m_res['kw_found']}/{m_res['kw_total']})"
                report_md += f"| {key} | {m_res['len']} char | {m_res['cer_raw']*100:.2f}% | {m_res['cer_norm']*100:.2f}% | {m_res['wer_raw']*100:.2f}% | {m_res['wer_norm']*100:.2f}% | {kw_str} |\n"
            else:
                report_md += f"| {key} | *N/A* | - | - | - | - | - |\n"
        report_md += "\n"

    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n[SUCCESS] Laporan lengkap -> {report_path}")


if __name__ == "__main__":
    main()
