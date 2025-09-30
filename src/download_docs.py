import os
from langchain_community.document_loaders import SitemapLoader

def save_focused_docs(section="chains", limit=20):
    os.makedirs("data", exist_ok=True)

    # Load docs from LangChain sitemap
    sitemap_url = "https://python.langchain.com/sitemap.xml"
    loader = SitemapLoader(sitemap_url)
    docs = loader.load()

    print(f"Total docs fetched: {len(docs)}")

    # Filter docs by section keyword in URL
    focused_docs = [doc for doc in docs if section in doc.metadata["source"]]

    print(f"Docs matching '{section}': {len(focused_docs)}")

    # Limit to 20 (or fewer if not enough exist)
    focused_docs = focused_docs[:limit]

    for i, doc in enumerate(focused_docs):
        filename = f"data/{section}_doc_{i}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(doc.page_content)

        # Save metadata (URL, etc.)
        meta_file = f"data/{section}_doc_{i}_meta.txt"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(str(doc.metadata))

    print(f"✅ Saved {len(focused_docs)} '{section}' docs into 'data/' folder.")

if __name__ == "__main__":
    # Example: focus on "chains" section, limit 20 pages
    save_focused_docs(section="chains", limit=20)