import os
import re
import pandas as pd
import logging
import openpyxl

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_log_file(file_path):
    """Parses a baseline_metrics.log or cascaded log to extract metrics."""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Bộ metric dùng chung (ViHateT5): Acc, WF1, MF1 — dòng dạng "Acc     | 0.9036"
        patterns = {
            'Cls Accuracy': r"(?:Cls Accuracy|(?<!Token\s)(?<!Syllable\s)(?<!Char\s)Accuracy).*?[|:]\s*([\d\.]+)",
            'Cls F1-Macro': r"(?:Cls F1-Macro|Macro F1).*?[|:]\s*([\d\.]+)",
            'Cls Precision (Macro)': r"(?:Cls Precision \(Macro\)|Precision \(Macro\)).*?[|:]\s*([\d\.]+)",
            'Cls Recall (Macro)': r"(?:Cls Recall \(Macro\)|Recall \(Macro\)).*?[|:]\s*([\d\.]+)",
            # Span chuẩn Table 3
            'Acc': r"\bAcc\s+[|:]\s*([\d\.]+)",
            'WF1': r"\bWF1\s+[|:]\s*([\d\.]+)",
            'MF1': r"\bMF1\s+[|:]\s*([\d\.]+)",
            # E1 format (Char Acc, Char WF1, Char MF1)
            'Char Acc': r"Char Acc.*?[|:]\s*([\d\.]+)",
            'Char WF1': r"Char WF1.*?[|:]\s*([\d\.]+)",
            'Char MF1 (Macro)': r"(?:Char MF1 \(Macro\)|Char MF1).*?[|:]\s*([\d\.]+)",
            'Token Accuracy': r"Token Accuracy.*?[|:]\s*([\d\.]+)",
            'Token mF1': r"Token mF1.*?[|:]\s*([\d\.]+)",
            'Syllable Accuracy': r"Syllable Accuracy.*?[|:]\s*([\d\.]+)",
            'Syllable F1 (Macro)': r"Syllable F1 \(Macro\).*?[|:]\s*([\d\.]+)",
            'Span IoU': r"(?:Span IoU F1|Span IoU|Span F1 \(IoU >= 0.5\)).*?[|:]\s*([\d\.]+)",
            'Strict IoU F1 (Char-based F1)': r"(?:Strict IoU F1 \(Char-based F1\)|Char-based F1).*?[|:]\s*([\d\.]+)",
            'Token F1 (Pos)': r"(?:Token F1 \(Pos\)|Token F1 \(Positive Class\)).*?[|:]\s*([\d\.]+)",
            'Token AUPRC': r"Token AUPRC.*?[|:]\s*([\d\.]+)",
            'Faithful. Comp': r"(?:Faithful\. Comp|Comprehensiveness).*?[|:]\s*([\d\.]+)",
            'Faithful. Suff': r"(?:Faithful\. Suff|Sufficiency).*?[|:]\s*([\d\.]+)",
            'GMB-Subgroup AUC': r"GMB-Subgroup AUC.*?[|:]\s*([\d\.]+)",
            'GMB-BPSN': r"GMB-BPSN.*?[|:]\s*([\d\.]+)",
            'GMB-BNSP': r"GMB-BNSP.*?[|:]\s*([\d\.]+)",
            'AUROC': r"(?:AUROC)(?!.*GMB).*?[|:]\s*([\d\.]+)",
        }

        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                val = float(matches[-1])
                metrics[metric_name] = val

        # Chuẩn hóa: nếu có Char Acc/Char WF1/Char MF1 (E1) mà chưa có Acc/WF1/MF1 thì gán sang
        if 'Acc' not in metrics and 'Char Acc' in metrics:
            metrics['Acc'] = metrics['Char Acc']
        if 'WF1' not in metrics and 'Char WF1' in metrics:
            metrics['WF1'] = metrics['Char WF1']
        if 'MF1' not in metrics and 'Char MF1 (Macro)' in metrics:
            metrics['MF1'] = metrics['Char MF1 (Macro)']

        return metrics

    except FileNotFoundError:
        logging.warning(f"Log file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return None


def parse_e1_summary(file_path):
    """Parse E1 baseline summary (e1_standalone_results_summary.txt)."""
    metrics = parse_log_file(file_path)
    if metrics:
        metrics['Model'] = 'RATeD E1 (Baseline)'
        metrics['Type'] = 'VN_E1'
    return metrics

def main():
    BASE_VN_DIR = r"c:\Projects\RATeD-V\experiments\vietnamese\baseline"
    PROPOSED_VN_DIR = r"c:\Projects\RATeD-V\experiments\vietnamese\proposed\results"
    
    BASE_EN_DIR = r"c:\Projects\RATeD-V\experiments\english\baseline\results"
    PROPOSED_EN_DIR = r"c:\Projects\RATeD-V\experiments\english\proposed\results"
    ABLATION_EN_DIR = r"c:\Projects\RATeD-V\experiments\english\ablation_study\results"

    OUTPUT_FILE = r"c:\Projects\RATeD-V\reports\baseline_summary_v4.xlsx"

    data = []

    # Mapping to cleaner names
    name_mapping = {
        # Vietnamese
        "bert-base-multilingual-cased": "mBERT (cased)",
        "bert-base-multilingual-uncased": "mBERT (uncased)", 
        "distilbert-base-multilingual-cased": "Distil-mBERT",
        "vinai_phobert-base": "PhoBERT (Base)",
        "xlm-roberta-base": "XLM-RoBERTa (Base)",
        "RATeD_E1_baseline": "RATeD E1 (Baseline)",
        "vinai_phobert-base-v2": "PhoBERT (Base) V2",
        
        # New LLMs
        "gemini_2_5_flash_lite": "Gemini 2.5 Flash Lite",
        "gpt_4o_mini": "GPT-4o Mini",
        "Qwen2_5_7B_Instruct": "Qwen 2.5 7B Instruct",
        
        # English
        "RATeD_E1_baseline": "RATeD E1 (Baseline)",
        "bert-base-multilingual-cased": "mBERT (cased)",
        "bert-base-multilingual-uncased": "mBERT (uncased)",
        "distilbert-base-multilingual-cased": "Distil-mBERT",
        "xlm-roberta-base": "XLM-RoBERTa (Base)",
    }

    # --- 1. VIETNAMESE BASELINES (Token Classification) ---
    for root, dirs, files in os.walk(BASE_VN_DIR):
        # Look for baseline_metrics.log or baseline_*.log
        target_log = None
        if "baseline_metrics.log" in files:
            target_log = "baseline_metrics.log"
        else:
            # Look for dynamic LLM logs
            baseline_logs = [f for f in files if f.startswith("baseline_") and f.endswith(".log")]
            if baseline_logs:
                # Pick latest by name (usually contains timestamp) or mtime
                baseline_logs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
                target_log = baseline_logs[0]

        if target_log:
            curr_model_name = os.path.basename(root)
            if curr_model_name == "RATeD_E1_baseline":
                continue  # E1 báo cáo riêng qua parse_e1_summary
            log_path = os.path.join(root, target_log)
            model_name = name_mapping.get(curr_model_name, curr_model_name)
            logging.info(f"Processing VN Baseline: {model_name} (in {root}, file: {target_log})")
            metrics = parse_log_file(log_path)
            if metrics:
                metrics['Model'] = model_name
                metrics['Type'] = 'VN_Baseline'
                data.append(metrics)

    # --- 2. VIETNAMESE E1 (RATeD E1 Baseline) ---
    E1_SUMMARY = os.path.join(BASE_VN_DIR, "results", "RATeD_E1_baseline", "e1_standalone_results_summary.txt")
    if os.path.isfile(E1_SUMMARY):
        logging.info(f"Processing VN E1: {E1_SUMMARY}")
        metrics = parse_e1_summary(E1_SUMMARY)
        if metrics:
            data.append(metrics)

    # --- 3. VIETNAMESE PROPOSED (E11 & Ablation) ---
    possible_vn_dirs = [
        os.path.join(PROPOSED_VN_DIR, "cascaded"),
        os.path.join(PROPOSED_VN_DIR, "only_stage1"),
        os.path.join(PROPOSED_VN_DIR, "only_stage2"),
        PROPOSED_VN_DIR
    ]
    for vn_dir in possible_vn_dirs:
        if os.path.exists(vn_dir):
            logging.info(f"Scanning VN Proposed Results: {vn_dir}")
            files = os.listdir(vn_dir)
            # Sort by mtime (latest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(vn_dir, x)), reverse=True)
            
            for file in files:
                if not (file.endswith(".log") or file.endswith(".txt")):
                    continue
                
                log_path = os.path.join(vn_dir, file)
                folder_name = os.path.basename(vn_dir)
                
                # Determine Model Name based on filename
                if "judge_Gemini" in file or "judge_gemini" in file or "Gemini" in file:
                    judge_suffix = "Gemini"
                elif "specialist" in file:
                    judge_suffix = "Qwen Specialist"
                elif "local_qwen" in file or "local_Qwen" in file:
                    judge_suffix = "Qwen"
                else:
                    judge_suffix = "Unknown Judge"

                if folder_name == "cascaded":
                    model_name = f"RATeD-V E11 (VN - Cascaded - {judge_suffix})"
                elif folder_name == "only_stage1":
                    model_name = f"RATeD-V E11 (VN - Stage 1 Only - {judge_suffix})"
                elif folder_name == "only_stage2":
                    model_name = f"RATeD-V E11 (VN - Stage 2 Only - {judge_suffix})"
                elif "cascaded_results_" in file:
                    # Fallback for root folder files
                    clean_name = file.replace("cascaded_results_vn_judge_", "").replace("cascaded_results_", "").split("_202")[0]
                    model_name = f"RATeD-V E11 (VN - {clean_name})"
                else:
                    continue

                metrics = parse_log_file(log_path)
                # Check uniqueness by Model Name
                if metrics and not any(d.get('Model') == model_name and d.get('Type').startswith('VN_') for d in data):
                     logging.info(f"Adding VN Proposed: {model_name} from {file}")
                     metrics['Model'] = model_name
                     metrics['Type'] = 'VN_Proposed'
                     data.append(metrics)

    # --- 3. ENGLISH PROPOSED ---
    possible_en_dirs = [
        os.path.join(PROPOSED_EN_DIR, "cascaded"),
        os.path.join(PROPOSED_EN_DIR, "only_stage1"),
        os.path.join(PROPOSED_EN_DIR, "only_stage2"),
        PROPOSED_EN_DIR, 
        r"c:\Projects\RATeD-V\experiments\english\results"
    ]
    
    def detect_en_judge(filepath):
        """Read first few lines to detect Judge."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                head = [next(f) for _ in range(25)]
            content = "".join(head).upper()
            if "GEMINI" in content: return "Gemini"
            if "QWEN" in content: return "Qwen Specialist"
            if "GPT" in content: return "GPT-4"
            if "LOCAL" in content: return "Qwen Specialist" # Default assumption if Local
            return "Unknown"
        except:
            return "Unknown"

    for en_dir in possible_en_dirs:
        if os.path.exists(en_dir):
            logging.info(f"Scanning EN Results: {en_dir}")
            files = os.listdir(en_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(en_dir, x)), reverse=True)
            
            for file in files:
                if not (file.endswith(".log") or file.endswith(".txt")):
                    continue
                
                log_path = os.path.join(en_dir, file)
                folder_name = os.path.basename(en_dir)
                
                # Detect Judge for English logs which might have generic names
                judge_suffix = detect_en_judge(log_path)
                
                if folder_name == "cascaded":
                     model_name = f"RATeD-V E11 (EN - Cascaded - {judge_suffix})"
                elif folder_name == "only_stage1":
                     # Stage 1 is judge-agnostic technically, but usually run with specific judge in pipeline
                     model_name = f"RATeD-V E11 (EN - Stage 1 Only - {judge_suffix})" 
                elif folder_name == "only_stage2":
                     model_name = f"RATeD-V E11 (EN - Stage 2 Only - {judge_suffix})"
                elif "cascaded_results_en_" in file:
                     model_name = f"RATeD-V E11 (EN - Full - {judge_suffix})"
                elif "_metrics.log" in file:
                     clean_name = file.replace("cascaded_results_", "").replace("_metrics.log", "")
                     model_name = f"RATeD-V (EN - {clean_name})"
                else:
                    continue
                
                # Correction for Stage 1: It is same for all judges usually, so we might want to deduplicate or just keep one
                # But let's keep specific to verify consistency
                
                metrics = parse_log_file(log_path)
                # Allow adding different judge variants even if same folder type
                if metrics and not any(d.get('Model') == model_name and d.get('Type') == 'EN_Proposed' for d in data):
                     logging.info(f"Adding EN Proposed: {model_name} from {file}")
                     metrics['Model'] = model_name
                     metrics['Type'] = 'EN_Proposed'
                     data.append(metrics)

    # --- 4. ENGLISH BASELINES ---
    # Use os.walk similar to VN to catch all folders
    if os.path.exists(BASE_EN_DIR):
        logging.info(f"Scanning EN Baselines: {BASE_EN_DIR}")
        for root, dirs, files in os.walk(BASE_EN_DIR):
            # Prioritize full metrics log if available
            target_log = None
            if "baseline_metrics_full.log" in files:
                target_log = "baseline_metrics_full.log"
            elif "baseline_metrics.log" in files:
                target_log = "baseline_metrics.log"
            else:
                # Look for dynamic LLM logs
                baseline_logs = [f for f in files if f.startswith("baseline_") and f.endswith(".log")]
                if baseline_logs:
                    baseline_logs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
                    target_log = baseline_logs[0]
            
            if target_log:
                log_path = os.path.join(root, target_log)
                
                curr_model_name = os.path.basename(root)
                model_name = name_mapping.get(curr_model_name, curr_model_name)
                
                logging.info(f"Processing EN Baseline: {model_name} (in {root}, file: {target_log})")
                metrics = parse_log_file(log_path)
                if metrics:
                    metrics['Model'] = model_name
                    metrics['Type'] = 'EN_Baseline'
                    # Check if already added (some dirs might have multiple log aliases)
                    if not any(d.get('Model') == model_name and d.get('Type') == 'EN_Baseline' for d in data):
                        data.append(metrics)

    # --- 5. ENGLISH ABLATION STUDY ---
    if os.path.exists(ABLATION_EN_DIR):
        logging.info(f"Scanning EN Ablation: {ABLATION_EN_DIR}")
        for root, dirs, files in os.walk(ABLATION_EN_DIR):
             # Ablation logs usually named ablation_metrics_NoFusion.log or just ablation_metrics.log
             for file in files:
                valid_log = False
                if file.endswith(".log"):
                    if file.startswith("ablation_metrics") or file.startswith("ablation_results"):
                        valid_log = True
                
                if valid_log:
                    log_path = os.path.join(root, file)
                    
                    # Extract variant name from filename or folder name
                    # Filename format: ablation_metrics_NoFusion.log or ablation_results_NoRAG_local.log
                    clean_name = file.replace(".log", "")
                    if "ablation_metrics_" in clean_name:
                        variant = clean_name.replace("ablation_metrics_", "")
                    elif "ablation_results_" in clean_name:
                        variant = clean_name.replace("ablation_results_", "")
                        # Remove provider/extra info if needed used in filename (e.g. "_local")
                        # e.g. NoRAG_local -> NoRAG
                        if "_local" in variant: variant = variant.replace("_local", "")
                        if "_gemini" in variant: variant = variant.replace("_gemini", "")
                    else:
                        # Fallback to folder name
                        folder_name = os.path.basename(root)
                        variant = folder_name
                        
                    model_name = f"RATeD Ablation ({variant})"
                    
                    logging.info(f"Processing EN Ablation: {model_name} (in {root})")
                    metrics = parse_log_file(log_path)
                    if metrics:
                        metrics['Model'] = model_name
                        metrics['Type'] = 'EN_Ablation'
                        data.append(metrics)

    if not data:
        logging.warning("No data collected. Check if logs exist.")
        return

    # Define Column Structure (Multi-index)
    # Define Column Structures
    # VN: đúng 7 chỉ số có trong log (Cls + Acc/WF1/MF1)
    vn_columns = [
        ("Model", ""),
        ("Classification", "Accuracy"),
        ("Classification", "F1-Macro"),
        ("Classification", "Precision"),
        ("Classification", "Recall"),
        ("Span Detection", "Acc"),
        ("Span Detection", "WF1"),
        ("Span Detection", "MF1"),
    ]
    
    en_columns = [
        ("Model", ""),
        # GROUP 1: CLASSIFICATION PERFORMANCE
        ("CLASSIFICATION PERFORMANCE", "Accuracy"),
        ("CLASSIFICATION PERFORMANCE", "F1-Macro"),
        # ("CLASSIFICATION PERFORMANCE", "Precision"),
        # ("CLASSIFICATION PERFORMANCE", "Recall"),
        ("CLASSIFICATION PERFORMANCE", "AUROC"),
        
        # GROUP 2: Bias (Fairness)
        ("Bias (Fairness)", "GMB-Subgroup AUC"),
        ("Bias (Fairness)", "GMB-BPSN"),
        ("Bias (Fairness)", "GMB-BNSP"),
        
        # GROUP 3: Explainability (Plausibility)
        ("Explainability (Plausibility)", "IOU-F1"),
        ("Explainability (Plausibility)", "Token F1 (Pos)"),
        ("Explainability (Plausibility)", "AUPRC"),
        
        # GROUP 4: Faithfulness
        ("Faithfulness", "Comprehensiveness (High is good)"),
        ("Faithfulness", "Sufficiency (Low is good)"),
    ]
    
    # helper to create df
    def create_df(rows, cols_schema, region="VN"):
        if not rows: return pd.DataFrame()
        
        # Prepare data list directly
        data_rows = []
        for row in rows:
            if region == "VN":
                new_row = [
                    row.get('Model'),
                    row.get('Cls Accuracy'),
                    row.get('Cls F1-Macro'),
                    row.get('Cls Precision (Macro)'),
                    row.get('Cls Recall (Macro)'),
                    row.get('Acc'),
                    row.get('WF1'),
                    row.get('MF1'),
                ]
            else: # EN
                new_row = [
                    row.get('Model'),
                    row.get('Cls Accuracy'),
                    row.get('Cls F1-Macro'),
                    # row.get('Cls Precision (Macro)'),
                    # row.get('Cls Recall (Macro)'),
                    row.get('AUROC'),
                    
                    row.get('GMB-Subgroup AUC'),
                    row.get('GMB-BPSN'),
                    row.get('GMB-BNSP'),
                    
                    row.get('Span IoU'),
                    row.get('Token F1 (Pos)'),
                    row.get('Token AUPRC'),
                    
                    row.get('Faithful. Comp'),
                    row.get('Faithful. Suff')
                ]
            data_rows.append(new_row)
            
        final_df = pd.DataFrame(data_rows, columns=pd.MultiIndex.from_tuples(cols_schema))
        return final_df

    # Split Data
    vn_data = [d for d in data if d['Type'].startswith("VN")]
    en_data = [d for d in data if d['Type'].startswith("EN_") and d['Type'] != 'EN_Ablation']
    
    # Ablation: Explicitly grabbing 'Full' models for reference
    ablation_data = [d for d in data if d['Type'] == 'EN_Ablation']
    
    # Grab references from Proposed data
    refs = [
        ("RATeD-V (EN - Local Final)", "RATeD-V E11 (Full - Local)"),
        ("RATeD-V (EN - Gemini Final)", "RATeD-V E11 (Full - Gemini)")
    ]
    
    for orig_name, new_name in reversed(refs): # Use reversed to maintain order when inserting at index 0
        matching = [d for d in data if d.get('Model') == orig_name]
        if matching:
            ref_copy = matching[0].copy()
            ref_copy['Model'] = new_name
            ablation_data.insert(0, ref_copy) # Put at top

    vn_df = create_df(vn_data, vn_columns, region="VN")
    en_df = create_df(en_data, en_columns, region="EN")
    ablation_df = create_df(ablation_data, en_columns, region="EN") # Reuse EN columns for Ablation

    # Save to Excel with Multiple Sheets
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            if not vn_df.empty:
                vn_df.to_excel(writer, sheet_name='Vietnamese Results')
                # Auto-adjust column width (Simple approach)
                # worksheet = writer.sheets['Vietnamese Results']
                # for column_cells in worksheet.columns:
                #     length = max(len(str(cell.value)) for cell in column_cells)
                #     worksheet.column_dimensions[column_cells[0].column_letter].width = length
            
            if not en_df.empty:
                en_df.to_excel(writer, sheet_name='English Results')
                
            if not ablation_df.empty:
                ablation_df.to_excel(writer, sheet_name='Ablation Study')
            
        logging.info(f"Successfully wrote report to {OUTPUT_FILE} (Sheets: Vietnamese Results, English Results, Ablation Study)")
    except Exception as e:
        logging.error(f"Failed to write Excel file: {e}")

if __name__ == "__main__":
    main()
