import logging
from rdflib import Graph, URIRef

# 配置你的 Graph 路径
KG_PATH = "/space_mounts/atai-hs25/dataset/graph.nt"

def check_movie_genres(g, movie_name, movie_id):
    print(f"\n--- Checking Genres for {movie_name} ({movie_id}) ---")
    
    # 查询该电影的所有属性，看看它是怎么被描述的
    query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?pLabel ?oLabel ?p ?o
        WHERE {{
            wd:{movie_id} ?p ?o .
            OPTIONAL {{ ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel) = "en") }}
            OPTIONAL {{ ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "en") }}
        }}
    """
    
    results = g.query(query)
    count = 0
    for row in results:
        p_lbl = row.pLabel if row.pLabel else str(row.p).split('/')[-1]
        o_lbl = row.oLabel if row.oLabel else str(row.o).split('/')[-1]
        
        # 重点关注 P136 (Genre) 或者其他分类属性
        if "P136" in str(row.p) or "genre" in str(p_lbl).lower():
            print(f"FOUND GENRE: {p_lbl} -> {o_lbl}")
        else:
            # 打印前5个其他属性，看看有没有异常
            if count < 5: 
                print(f"Property: {p_lbl} -> {o_lbl}")
        count += 1
            
    if count == 0:
        print("❌ No properties found for this entity!")

def main():
    print(f"Loading Graph from {KG_PATH} ...")
    try:
        g = Graph()
        g.parse(KG_PATH, format="nt")
        print(f"Graph loaded. {len(g)} triples.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. 检查狮子王 (Q36479)
    check_movie_genres(g, "The Lion King", "Q36479")
    
    # 2. 检查索多玛120天 (Q657977) - 注意：你需要确认这个ID，我是从wikidata查的，你的库里可能是别的
    # 你的日志里没显示Salò的ID，如果能看到日志里的ID最好替换一下
    
    # 3. 检查 Pocahontas (Q193406)
    check_movie_genres(g, "Pocahontas", "Q193406")

if __name__ == "__main__":
    main()