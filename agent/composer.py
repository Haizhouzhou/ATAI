from agent.constants import PREFIXES

class Composer:
    def compose_one_hop_qa(self, uri, pred, direct):
        if direct == "forward":
            return f"{PREFIXES} SELECT ?o ?oLabel WHERE {{ <{uri}> <{pred}> ?o . OPTIONAL {{ ?o rdfs:label ?oLabel . }} }}"
        return f"{PREFIXES} SELECT ?s ?sLabel WHERE {{ ?s <{pred}> <{uri}> . OPTIONAL {{ ?s rdfs:label ?sLabel . }} }}"

    def compose_graph_rec_query(self, seed):
        return f"""
            {PREFIXES} SELECT DISTINCT ?movie ?rating WHERE {{
                {{ <{seed}> wdt:P136 ?g . ?movie wdt:P136 ?g . }} UNION
                {{ <{seed}> wdt:P57 ?d . ?movie wdt:P57 ?d . }}
                ?movie wdt:P31 wd:Q11424 . OPTIONAL {{ ?movie ddis:rating ?rating . }}
            }} LIMIT 50
        """