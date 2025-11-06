import argparse
from datetime import datetime
from .config import CSV_PATH, N_ROWS
from .dataio import load_df
from .anonymize import anonymize_text
from .pii_patterns import PII_LABELS
from .utils_text import short

def main(csv: str, n_rows: int, use_labels: bool):
    dfN = load_df(csv, n_rows)

    counts_global, anon_list = {}, []
    for _, row in dfN.iterrows():
        local = {}
        anon = anonymize_text(row.get("comment_text",""), local, use_label_tokens=use_labels)
        anon_list.append(anon)
        for k, v in local.items():
            counts_global[k] = counts_global.get(k, 0) + v

    dfN["comment_text_anonymized"] = anon_list

    print("\nAperçu anonymisé (5 lignes) :")
    print(dfN.loc[:, ["id","comment_text_anonymized"]].head(5).to_string(index=False))

    print("\nComparaison (10 lignes) :")
    prev = dfN.loc[:, ["id","comment_text","comment_text_anonymized"]].copy()
    prev["comment_text"] = prev["comment_text"].apply(lambda s: short(s, 120))
    prev["comment_text_anonymized"] = prev["comment_text_anonymized"].apply(lambda s: short(s, 120))
    print(prev.head(10).to_string(index=False))

    print("\n=== REGISTRE (imprimé) — Anonymisation DCP uniquement ===")
    print(f"Date UTC         : {datetime.utcnow().isoformat()}Z")
    print(f"Fichier traité   : {csv} (N_ROWS={len(dfN)})")
    print("Modèle NER       : spaCy en_core_web_sm (PERSON uniquement)")
    print("Catégories DCP   : PERSON, EMAIL, PHONE, ADDRESS, IP, URL, USERNAME, CREDIT_CARD")
    print(f"Sortie/format    : {'Labels (PERSON/EMAIL/...)' if use_labels else '**** (masquage intégral)'}")

    print("\nComptage DCP :")
    for k in ["PERSON","EMAIL","PHONE","ADDRESS","IP","URL","USERNAME","CREDIT_CARD"]:
        if k in counts_global:
            print(f"  - {k:>11}: {counts_global[k]}")

    # Sauvegarde rapide si tu veux réutiliser directement en étape 2
    dfN.to_csv("data/_anonymized_head.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--mask-labels", action="store_true", help="PERSON/EMAIL/... ; sinon ****")
    args = ap.parse_args()
    main(csv=args.csv, n_rows=args.n_rows, use_labels=args.mask_labels)
