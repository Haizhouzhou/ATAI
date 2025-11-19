import logging
import sys
from rdflib import Graph

# 配置你的 Graph 路径 (确保路径和你的 Config 中一致)
KG_PATH = "/space_mounts/atai-hs25/dataset/graph.nt"

def check_specific_movie(g, movie_id, movie_name):
    """查询特定电影是否有图片"""
    print(f"\n--- Checking {movie_name} ({movie_id}) ---")
    
    query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        
        SELECT ?p ?o
        WHERE {{
            wd:{movie_id} ?p ?o .
            FILTER (?p = wdt:P18)
        }}
    """
    
    results = g.query(query)
    found = False
    for row in results:
        print(f"✅ FOUND IMAGE URL: {row.o}")
        found = True
    
    if not found:
        print(f"❌ NO IMAGE found for {movie_name} ({movie_id})")

def find_any_movies_with_images(g, limit=5):
    """随机查找数据库中任意带有图片的电影"""
    print(f"\n--- Searching for ANY movies with images (Limit {limit}) ---")
    
    query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?movie ?label ?image
        WHERE {{
            ?movie wdt:P18 ?image .
            OPTIONAL {{ ?movie rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT {limit}
    """
    
    results = g.query(query)
    if not results:
        print("❌ CRITICAL: No images found in the entire dataset!")
    else:
        for row in results:
            label = row.label if row.label else "Unknown Label"
            print(f"🎬 Movie: {label} | ID: {row.movie} | Image: {row.image}")

def main():
    print(f"Loading Graph from {KG_PATH} ... (This may take a minute)")
    try:
        g = Graph()
        # 注意：你的文件后缀是 .nt，格式必须指定为 'nt'
        g.parse(KG_PATH, format="nt")
        print(f"Graph loaded successfully. Total triples: {len(g)}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 1. 检查 "Back to the Future" (电影版 ID Q91540)
    check_specific_movie(g, "Q91540", "Back to the Future (Movie)")

    # 2. 检查 "Back to the Future: The Game" (游戏版 ID Q91419 - 之前推荐错误的那个)
    check_specific_movie(g, "Q91419", "Back to the Future (Game)")

    # 3. 检查 "True Lies" (之前的推荐 Q110397)
    check_specific_movie(g, "Q110397", "True Lies")

    # 4. 找出数据库里到底哪些电影有图（用来做 Golden Sample）
    find_any_movies_with_images(g, limit=10)

if __name__ == "__main__":
    main()