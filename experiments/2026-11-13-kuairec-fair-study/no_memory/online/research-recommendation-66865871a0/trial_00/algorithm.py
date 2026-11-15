def rank(groups, context):
    return {str(g["id"]): [r["item"] for r in g["candidates"]] for g in groups}
