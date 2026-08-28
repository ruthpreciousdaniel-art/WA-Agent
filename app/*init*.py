# %% Extra cell: keep empty folders trackable in git
open("rag-agent/data/uploads/.gitkeep", "w").close()
open("rag-agent/data/faiss_index/.gitkeep", "w").close()
print("Added .gitkeep placeholders — now git will track these empty dirs.")
