"""
postprocessing.py — Modul Post-Processing Teks untuk OCR Pipeline

Modul ini menyediakan fungsi post-processing untuk memperbaiki
hasil OCR, khususnya untuk teks pseudocode berbahasa Indonesia.
Fungsi utama:
  - fix_keywords: Fuzzy matching keyword pseudocode
  - clean_ocr_artifacts: Bersihkan karakter noise dari OCR
  - detect_hallucination: Deteksi baris hallucination (TrOCR)
  - reconstruct_indentation: Rekonstruksi indentasi dari koordinat visual
  - postprocess_text: Pipeline post-processing lengkap
"""

import re
from typing import List, Dict, Optional, Tuple


# ============================================================================
# KAMUS KEYWORD PSEUDOCODE
# ============================================================================

PSEUDOCODE_KEYWORDS = [
    # Struktur program
    "function", "endfunction", "procedure", "endprocedure",
    "program", "endprogram",
    # Seksi
    "kamus", "algoritma",
    # Control flow
    "if", "then", "else", "endif",
    "while", "do", "endwhile",
    "for", "to", "endfor", "downto",
    "repeat", "until",
    "case", "of", "endcase", "otherwise",
    # I/O dan return
    "input", "output", "return", "print", "write", "read",
    # Tipe data
    "integer", "real", "string", "boolean", "char", "array",
    "true", "false",
    # Operator
    "not", "and", "or", "mod", "div",
    # Keyword umum pseudocode Indonesia
    "in", "out", "type",
    # Komentar
]

# Keyword yang sering salah baca oleh OCR, beserta pola salah umumnya
COMMON_OCR_MISTAKES = {
    # PaddleOCR common mistakes
    "seal": "real",
    "Seal": "real",
    "teal": "real",
    "reai": "real",
    "leal": "real",
    "1nteger": "integer",
    "lnteger": "integer",
    "intger": "integer",
    "integet": "integer",
    "lnput": "input",
    "1nput": "input",
    "0utput": "output",
    "retum": "return",
    "retrn": "return",
    "retun": "return",
    "retuin": "return",
    "algoritma": "algoritma",
    "algortima": "algoritma",
    "algortma": "algoritma",
    "algoritm": "algoritma",
    "algcritma": "algoritma",
    "lgoritma": "algoritma",
    "algorima": "algoritma",
    "aigorima": "algoritma",
    "aisonite": "algoritma",
    "algorithms": "algoritma",
    "algorithm": "algoritma",
    "endtunction": "endfunction",
    "endfunctior": "endfunction",
    "endfuncton": "endfunction",
    "endtuncton": "endfunction",
    "erkrunetion": "endfunction",
    "functon": "function",
    "tunction": "function",
    "funciton": "function",
    "funotion": "function",
    "funcion": "function",
    "1oo": "100",
    "1OO": "100",
    "IOO": "100",
    "amus": "kamus",
    "kam": "kamus",
    "skam": "kamus",
    "kamis": "kamus",
    "booiean": "boolean",
    "booleon": "boolean",
    "boolen": "boolean",
    "strlng": "string",
    "strng": "string",
    "sting": "string",
    "vhile": "while",
    "whlle": "while",
    "whi1e": "while",
    "endvhile": "endwhile",
    "endwhlle": "endwhile",
}

# Kata bahasa Inggris yang menandakan hallucination TrOCR
HALLUCINATION_INDICATORS = [
    "parliament", "robert", "schleibert", "wikipedia", "was", "been",
    "the", "which", "where", "would", "could", "should", "with",
    "have", "this", "that", "from", "into", "they", "their", "were",
    "very", "possible", "development", "education", "general",
    "history", "century", "country", "government", "university",
    "through", "vaughan", "civilian", "germanism", "fitzgerald",
    "bernardino", "barbara", "press", "labour", "productivist",
    "federal", "export", "known", "within", "hospital",
]


# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Hitung Levenshtein distance antara dua string."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_closest_keyword(word: str, max_distance: int = 2) -> Optional[str]:
    """
    Cari keyword pseudocode yang paling mirip dengan word.
    
    Args:
        word: Kata yang akan dicocokkan
        max_distance: Levenshtein distance maksimum
    
    Returns:
        Keyword terdekat, atau None jika tidak ditemukan yang cukup dekat
    """
    word_lower = word.lower()
    
    # Exact match check
    if word_lower in PSEUDOCODE_KEYWORDS:
        return word_lower
    
    # Cek common mistakes dulu (prioritas tinggi)
    if word in COMMON_OCR_MISTAKES:
        return COMMON_OCR_MISTAKES[word]
    if word_lower in COMMON_OCR_MISTAKES:
        return COMMON_OCR_MISTAKES[word_lower]
    
    # Fuzzy match terhadap keyword list
    best_match = None
    best_distance = max_distance + 1
    
    for keyword in PSEUDOCODE_KEYWORDS:
        dist = levenshtein_distance(word_lower, keyword)
        if dist < best_distance:
            best_distance = dist
            best_match = keyword
    
    if best_distance <= max_distance:
        return best_match
    
    return None


# ============================================================================
# FUNGSI POST-PROCESSING UTAMA
# ============================================================================

def fix_keywords(text: str, max_distance: int = 2) -> str:
    """
    Perbaiki kata-kata dalam teks menggunakan fuzzy matching
    terhadap kamus keyword pseudocode.
    
    Hanya kata-kata yang mirip keyword (Levenshtein ≤ max_distance) yang diperbaiki.
    Kata-kata lain (variabel, angka, operator) tidak diubah.
    
    Args:
        text: Teks OCR yang akan diperbaiki
        max_distance: Jarak Levenshtein maksimum untuk dianggap cocok
    
    Returns:
        Teks dengan keyword yang diperbaiki
    """
    # Cek common mistakes di level string dulu (untuk multi-word patterns)
    for mistake, correction in COMMON_OCR_MISTAKES.items():
        if mistake in text:
            text = text.replace(mistake, correction)
    
    # Proses per kata
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Pisahkan leading whitespace (indentasi)
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        if not stripped:
            fixed_lines.append(line)
            continue
        
        # Split dengan mempertahankan separator
        # Pattern: kata-kata dipisahkan oleh spasi, operator, tanda baca
        tokens = re.split(r'(\s+|[(),:;=+\-*/\[\]{}<>])', stripped)
        fixed_tokens = []
        
        for token in tokens:
            if not token or token.isspace() or len(token) <= 1:
                fixed_tokens.append(token)
                continue
            
            # Skip jika token adalah angka atau operator
            if token.replace('.', '').replace('-', '').isdigit():
                fixed_tokens.append(token)
                continue
            
            # Coba match keyword
            match = find_closest_keyword(token, max_distance)
            if match:
                fixed_tokens.append(match)
            else:
                fixed_tokens.append(token)
        
        fixed_line = indent + ''.join(fixed_tokens)
        fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)


def clean_ocr_artifacts(text: str) -> str:
    """
    Bersihkan artifacts umum dari OCR output.
    
    Menghapus/memperbaiki:
    - Karakter noise di awal baris (misalnya '(' dari TrOCR)
    - Spasi berlebih
    - Titik/koma yang tidak seharusnya
    """
    lines = text.split('\n')
    cleaned = []
    
    for line in lines:
        # Hapus prefix artifacts umum dari TrOCR
        # Pattern: baris dimulai dengan '(' atau '#' diikuti spasi, kemungkinan artifact
        stripped = line.strip()
        
        # Hapus leading '(' yang merupakan artifact, bukan bagian dari kode
        # Hanya hapus jika '(' di awal baris dan tidak diikuti oleh identifier/ekspresi
        if stripped.startswith('( ') and not re.match(r'^\(\s*\w+\s*[=<>+\-*/]', stripped):
            stripped = stripped[2:].strip()
            # Pertahankan indentasi original
            indent = line[:len(line) - len(line.lstrip())]
            line = indent + stripped
        
        # Hapus trailing '.' yang merupakan artifact, bukan bagian dari kode pseudocode
        if stripped.endswith(' .') and not stripped.endswith('..'):
            line = line.rstrip()
            if line.endswith(' .'):
                line = line[:-2]
        
        # Bersihkan multiple spaces (kecuali di awal baris untuk indentasi)
        indent = line[:len(line) - len(line.lstrip())]
        content = line.lstrip()
        content = re.sub(r'  +', ' ', content)
        line = indent + content
        
        cleaned.append(line)
    
    return '\n'.join(cleaned)


def detect_hallucination(line: str, threshold: int = 2) -> bool:
    """
    Deteksi apakah sebuah baris kemungkinan besar adalah hallucination.
    Berdasarkan keberadaan kata-kata bahasa Inggris non-keyword.
    
    Args:
        line: Baris teks untuk dicek
        threshold: Jumlah minimum kata hallucination untuk flag
    
    Returns:
        True jika baris kemungkinan hallucination
    """
    words = re.findall(r'[a-zA-Z]+', line.lower())
    
    hallucination_count = 0
    for word in words:
        if word in HALLUCINATION_INDICATORS:
            hallucination_count += 1
    
    return hallucination_count >= threshold


def filter_hallucinations(text: str, threshold: int = 2) -> str:
    """
    Filter/tandai baris-baris yang kemungkinan hallucination.
    Baris hallucination dihapus dari output.
    
    Args:
        text: Teks OCR lengkap
        threshold: Jumlah minimum kata hallucination per baris
    
    Returns:
        Teks tanpa baris hallucination
    """
    lines = text.split('\n')
    filtered = []
    
    for line in lines:
        if not detect_hallucination(line, threshold):
            filtered.append(line)
    
    return '\n'.join(filtered)


def reconstruct_indentation(transcribed_lines: List[str], 
                           line_boxes: List[Dict], 
                           global_min_x: float, 
                           avg_line_height: float,
                           space_factor: float = 0.5) -> List[str]:
    """
    Rekonstruksi indentasi berdasarkan posisi horizontal visual.
    
    Args:
        transcribed_lines: List teks per baris (tanpa indentasi)
        line_boxes: List dict dengan key 'xmin' untuk setiap baris
        global_min_x: Posisi x paling kiri dari semua baris
        avg_line_height: Rata-rata tinggi baris (pixel)
        space_factor: Faktor skala untuk menghitung lebar karakter
    
    Returns:
        List teks per baris dengan indentasi
    """
    indented = []
    char_width = avg_line_height * space_factor
    
    for text, box in zip(transcribed_lines, line_boxes):
        if not text.strip():
            indented.append(text)
            continue
        
        offset = max(0, box['xmin'] - global_min_x)
        num_spaces = int(round(offset / char_width)) if char_width > 0 else 0
        
        indented_text = (" " * num_spaces) + text.strip()
        indented.append(indented_text)
    
    return indented


def postprocess_text(text: str, 
                     do_fix_keywords: bool = True,
                     do_clean_artifacts: bool = True,
                     do_filter_hallucination: bool = False,
                     keyword_max_distance: int = 2,
                     hallucination_threshold: int = 2) -> str:
    """
    Pipeline post-processing lengkap (terintegrasi).
    Urutan: clean artifacts → filter hallucination → fix keywords.
    
    Args:
        text: Raw OCR output text
        do_fix_keywords: Aktifkan keyword fuzzy matching
        do_clean_artifacts: Aktifkan pembersihan artifact OCR
        do_filter_hallucination: Aktifkan filter hallucination (untuk TrOCR)
        keyword_max_distance: Max Levenshtein distance untuk keyword matching
        hallucination_threshold: Min kata hallucination per baris
    
    Returns:
        Post-processed text
    """
    result = text
    
    if do_clean_artifacts:
        result = clean_ocr_artifacts(result)
    
    if do_filter_hallucination:
        result = filter_hallucinations(result, threshold=hallucination_threshold)
    
    if do_fix_keywords:
        result = fix_keywords(result, max_distance=keyword_max_distance)
    
    return result
