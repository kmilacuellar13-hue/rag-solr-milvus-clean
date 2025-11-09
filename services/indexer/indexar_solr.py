import argparse, json, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solr", default="http://localhost:8983/solr/rag2")
    ap.add_argument("--input", default="data/corpus/corpus_texto.jsonl")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    docs, batch = [], args.batch

    def send(d):
        if not d: return
        r = requests.post(f"{args.solr}/update?commitWithin=1000&wt=json",
                          headers={"Content-Type":"application/json"},
                          data=json.dumps(d))
        r.raise_for_status()

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
            if len(docs) >= batch:
                send(docs); docs = []
    send(docs)

    # commit final
    requests.get(f"{args.solr}/update?commit=true&wt=json")
    # ping
    ping = requests.get(f"{args.solr}/admin/ping?wt=json").json()
    print("Ping:", ping)

if __name__ == "__main__":
    main()
