import pandas as pd
import numpy as np

# Load the new datasets
df_final = pd.read_csv('stats/pareto_final.csv')
df_evol = pd.read_csv('stats/pareto_evolucao.csv')

kcal_cols = [col for col in df_final.columns if col.endswith('_kcal')]

def check_errors(df, name):
    report = []
    report.append(f"=== Análise do arquivo: {name} ===")
    
    # 1. Check for negative kcal
    min_kcals = df[kcal_cols].min()
    neg_kcals = min_kcals[min_kcals < -1e-9] # using small tolerance
    if len(neg_kcals) > 0:
        report.append(f"⚠️ ERRO: Valores negativos encontrados:")
        report.append(str(neg_kcals))
    else:
        report.append(f"✅ Sucesso: Nenhum grupo alimentar possui valor de kcal negativo (Menor valor calórico gerado: {min_kcals.min():.6f}).")
        
    # 2. Check for DGV + REV > Vegetables
    veg_diff = df['vegetables_kcal'] - (df['dgv_kcal'] + df['rev_kcal'])
    min_veg_diff = veg_diff.min()
    if min_veg_diff < -1e-9: # small tolerance for float precision
        report.append(f"⚠️ ERRO: Violação de subgrupos (DGV + REV > Vegetables). Diferença mínima: {min_veg_diff:.6f}")
    else:
        report.append(f"✅ Sucesso: A regra de subgrupos (DGV + REV <= Vegetables) foi respeitada em todas as dietas (Folga mínima: {min_veg_diff:.6f} kcal).")
        
    # 3. Check fixed groups consistency
    fixed_groups = ['nuts_and_peanuts_kcal', 'whole_cereals_kcal', 'fish_and_seafood_kcal', 'dgv_kcal', 'refined_cereals_kcal']
    fixed_issues = False
    for fg in fixed_groups:
        std = df[fg].std()
        # std could be nan if only 1 row, but here we have many rows
        if pd.notna(std) and std > 1e-6:
            report.append(f"⚠️ ERRO: Grupo fixo {fg} não está constante! Desvio Padrão: {std}")
            fixed_issues = True
    if not fixed_issues:
        report.append("✅ Sucesso: Todos os 5 grupos com 'valor_fixo' permaneceram estritamente inalterados durante toda a execução.")
        
    return "\n".join(report)

print(check_errors(df_final, 'pareto_final.csv'))
print()
print(check_errors(df_evol, 'pareto_evolucao.csv'))