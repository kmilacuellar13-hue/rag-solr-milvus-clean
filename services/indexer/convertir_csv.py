import argparse, pandas as pd, json, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/corpus/corpus_bloques_100.csv")
    ap.add_argument("--output", default="data/corpus/corpus_texto.jsonl")
    ap.add_argument("--text-col", default="texto_limpio",
                    help="Nombre de la columna con el texto (por defecto: texto_limpio).")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.text_col not in df.columns:
      raise ValueError(f"No existe la columna '{args.text_col}' en el CSV.")

    # Generar id si no existe
    if "id" not in df.columns:
        df.insert(0, "id", [f"doc_{i:06d}" for i in range(len(df))])

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df["id"] = df["id"].astype(str)

    outp = pathlib.Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {"id": row["id"], "text": row[args.text_col]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"OK -> {args.output} ({len(df)} docs)")

if __name__ == "__main__":
    main()
