# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gerar_csv_grande(csv_path: str, n_rows: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    # Numericas (ASCII-only)
    idade = rng.integers(18, 75, size=n_rows)
    renda_mensal = rng.normal(5200, 2200, size=n_rows).clip(700, 30000)
    score_credito = rng.normal(620, 90, size=n_rows).clip(250, 900)
    tempo_emprego_meses = rng.integers(0, 360, size=n_rows)
    numero_compras_90d = rng.poisson(6, size=n_rows).clip(0, 60)
    tempo_site_min = rng.gamma(shape=2.0, scale=8.0, size=n_rows).clip(0, 240)
    taxa_reclamacoes = (rng.beta(2, 30, size=n_rows) * 10.0).clip(0, 10)
    atraso_pagamento_dias = rng.poisson(2, size=n_rows).clip(0, 60)

    # Categoricas
    cidade = rng.choice(
        ["Campinas", "Sao_Paulo", "Rio_de_Janeiro", "Belo_Horizonte", "Curitiba", "Salvador"],
        size=n_rows,
        p=[0.22, 0.28, 0.14, 0.12, 0.12, 0.12],
    )
    canal_aquisicao = rng.choice(
        ["organico", "ads", "indicacao", "email", "parceria"],
        size=n_rows,
        p=[0.30, 0.28, 0.18, 0.12, 0.12],
    )
    plano = rng.choice(["basico", "plus", "pro"], size=n_rows, p=[0.55, 0.30, 0.15])
    segmento = rng.choice(
        ["varejo", "industria", "servicos", "agro", "educacao"],
        size=n_rows,
        p=[0.35, 0.20, 0.25, 0.10, 0.10],
    )
    dispositivo = rng.choice(["mobile", "desktop", "tablet"], size=n_rows, p=[0.62, 0.33, 0.05])

    # Missings artificiais
    renda_mensal = renda_mensal.astype(float)
    renda_mensal[rng.random(n_rows) < 0.05] = np.nan
    cidade = cidade.astype(object)
    cidade[rng.random(n_rows) < 0.03] = None

    # Target binario
    renda_eff = np.nan_to_num(renda_mensal, nan=np.nanmedian(renda_mensal))
    logit = (
        0.018 * (idade - 40)
        + 0.00035 * (renda_eff - 5200)
        + 0.010 * (score_credito - 620)
        + 0.004 * (tempo_emprego_meses - 60)
        + 0.070 * (numero_compras_90d - 6)
        - 0.050 * (atraso_pagamento_dias)
        - 0.090 * (taxa_reclamacoes)
        + np.where(plano == "pro", 0.9, 0.0)
        + np.where(canal_aquisicao == "indicacao", 0.25, 0.0)
        + np.where(dispositivo == "mobile", 0.10, 0.0)
        - 1.3
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "idade": idade,
            "renda_mensal": renda_mensal,
            "score_credito": score_credito,
            "tempo_emprego_meses": tempo_emprego_meses,
            "numero_compras_90d": numero_compras_90d,
            "tempo_site_min": tempo_site_min,
            "taxa_reclamacoes": taxa_reclamacoes,
            "atraso_pagamento_dias": atraso_pagamento_dias,
            "cidade": cidade,
            "canal_aquisicao": canal_aquisicao,
            "plano": plano,
            "segmento": segmento,
            "dispositivo": dispositivo,
            "target": target,
        }
    )

    ensure_dir(os.path.dirname(csv_path) or ".")
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/dataset_grande.csv")
    parser.add_argument("--rows", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gerar_csv_grande(args.out, n_rows=args.rows, seed=args.seed)
    print(f"[OK] CSV gerado em: {os.path.abspath(args.out)}")
    print(f"[OK] Linhas: {args.rows}")
    print(f"[OK] Seed: {args.seed}")


if __name__ == "__main__":
    main()
